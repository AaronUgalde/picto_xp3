"""
build_dataset.py
----------------
Pasos 1-4 (+ exportación) para construir un corpus de instrucciones
texto(es) -> secuencia de IDs ARASAAC a partir de bigscience/xP3mt.

Uso típico:
    # prueba rápida con 50k registros
    python build_dataset.py --max_records 50000 --out_dir out_dev

    # corrida completa (~3.37M registros de es/merged_es.jsonl, ~7.8 GB)
    python build_dataset.py --num_proc 8 --spacy_model es_core_news_lg --out_dir out_full

Salida (out_dir/):
    train.jsonl, validation.jsonl, test.jsonl   (formato chat + campos de análisis)
    hf_dataset/                                 (DatasetDict.save_to_disk)
    dataset_stats.json                          (insights del Paso 5)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata

from datasets import Dataset, DatasetDict, load_dataset

from arasaac_lexicon import ArasaacLexicon
from text2picto_es import Text2PictoES

# --------------------------------------------------------------------------- #
# PASO 1 — Ingesta
# --------------------------------------------------------------------------- #
# En xP3mt la carpeta es/ contiene ~400 jsonl por tarea Y un merged_es.jsonl
# que ya los concatena. Usa UNO de los dos: si cargas "es/*.jsonl" duplicas
# todo el corpus.
DEFAULT_FILES = "es/merged_es.jsonl"
REPO = "hf://datasets/bigscience/xP3mt"


def load_xp3mt(data_files: str, max_records: int | None) -> Dataset:
    files = [f"{REPO}/{f.strip()}" for f in data_files.split(",")]
    if max_records:
        stream = load_dataset("json", data_files=files, split="train", streaming=True)

        def gen():
            for k, ex in enumerate(stream.take(max_records)):
                yield {"id": k, "inputs": ex["inputs"], "targets": ex["targets"]}
        return Dataset.from_generator(gen)
    ds = load_dataset("json", data_files=files, split="train")
    ds = ds.add_column("id", list(range(len(ds))))
    return ds.select_columns(["id", "inputs", "targets"])


# --------------------------------------------------------------------------- #
# PASO 2 — Limpieza y filtrado
# --------------------------------------------------------------------------- #
CODE_RE = re.compile(
    r"(```|</?\w+>|\bdef\s+\w+\(|\bfunction\s*\(|\breturn\b|[{};]\s*$|=>|::|\$\w+|"
    r"\bSELECT\b.*\bFROM\b|https?://|www\.|\\[a-z]+\{)", re.I | re.M)
LETTER_RE = re.compile(r"[a-záéíóúüñ]", re.I)
ES_CHARS = re.compile(r"[áéíóúñ¿¡]", re.I)
ES_STOP = {"el", "la", "de", "que", "y", "en", "los", "se", "del", "las", "un", "por",
           "con", "no", "una", "su", "para", "es", "al", "lo", "como", "más", "pero",
           "yo", "me", "él", "ella", "muy", "este", "esta", "hay", "ya", "también"}


def normalize_text(t: str) -> str:
    t = unicodedata.normalize("NFC", t or "")
    t = t.replace("​", " ").replace("\xa0", " ")
    t = re.sub(r"\s+", " ", t).strip()
    t = t.strip(" \"'“”«»")
    return t


def heuristic_ok(t: str, min_words: int, max_words: int) -> bool:
    if not t:
        return False
    words = t.split()
    if not (min_words <= len(words) <= max_words):
        return False
    if CODE_RE.search(t):
        return False
    letters = len(LETTER_RE.findall(t))
    if letters / max(len(t.replace(" ", "")), 1) < 0.70:       # ruido, símbolos, números
        return False
    if re.search(r"(.)\1{4,}", t):                               # "aaaaa", "!!!!!"
        return False
    if sum(w.isupper() and len(w) > 1 for w in words) / len(words) > 0.5:
        return False
    return True


class LangID:
    """fastText lid (NLLB, 218 idiomas). Distingue es de pt/ca/gl, que es
    donde fallan las heurísticas."""
    _model = None

    @classmethod
    def get(cls):
        if cls._model is None:
            import fasttext
            from huggingface_hub import hf_hub_download
            path = hf_hub_download("facebook/fasttext-language-identification", "model.bin")
            cls._model = fasttext.load_model(path)
        return cls._model

    @classmethod
    def spanish_prob(cls, texts):
        labels, probs = cls.get().predict([t.replace("\n", " ") for t in texts], k=1)
        return [float(p[0]) if l[0] == "__label__spa_Latn" else 0.0 for l, p in zip(labels, probs)]


def clean(ds: Dataset, args) -> Dataset:
    n0 = len(ds)
    ds = ds.map(lambda b: {"target_es": [normalize_text(t) for t in b["targets"]]},
                batched=True, num_proc=args.num_proc, desc="normalizar")
    ds = ds.filter(lambda b: [heuristic_ok(t, args.min_words, args.max_words) for t in b["target_es"]],
                   batched=True, num_proc=args.num_proc, desc="heurísticas")
    n1 = len(ds)

    def lid(b):
        try:
            p = LangID.spanish_prob(b["target_es"])
        except ImportError:   # respaldo si no hay fasttext: más débil
            p = [1.0 if (ES_CHARS.search(t) or
                         sum(w.lower() in ES_STOP for w in t.split()) >= 2) else 0.0
                 for t in b["target_es"]]
        return [x >= args.lid_threshold for x in p]
    # Un solo proceso a propósito: el modelo fastText pesa ~1.2 GB y cargarlo en
    # cada subproceso (spawn en macOS) los mata por memoria. fastText ya es muy
    # rápido (decenas de miles de oraciones/s), así que no se pierde casi nada.
    ds = ds.filter(lid, batched=True, batch_size=5000, desc="idioma=es")
    n2 = len(ds)

    # Deduplicación de pares (input, target) idénticos
    seen = set()

    def dedup(ex):
        h = hashlib.md5((ex["inputs"] + "\x00" + ex["target_es"]).encode()).hexdigest()
        if h in seen:
            return False
        seen.add(h)
        return True
    ds = ds.filter(dedup, num_proc=1, desc="dedup")
    n3 = len(ds)
    print(f"[paso 2] {n0:,} -> heurísticas {n1:,} -> idioma {n2:,} -> dedup {n3:,}")
    args._clean_counts = {"raw": n0, "heuristics": n1, "lang_es": n2, "dedup": n3}
    return ds


# --------------------------------------------------------------------------- #
# PASO 3 — Reformulación de prompts
# --------------------------------------------------------------------------- #
# Se conserva la tarea original de xP3mt tal cual; solo se sustituye la
# mención del idioma de salida ("Spanish"/"español") por "secuencia de
# pictogramas". No se añade ninguna instrucción nueva.
#   "Write a summary of the previous text in Spanish:"
#       -> "Write a summary of the previous text in a sequence of pictograms:"
#   "The same text in Spanish:" -> "The same text in a sequence of pictograms:"
#   "Resume el texto en español:" -> "Resume el texto en una secuencia de pictogramas:"
#
# Solo se reemplaza la ÚLTIMA mención: la instrucción va al final del prompt
# en casi todas las plantillas xP3, y así no se tocan menciones de "Spanish"
# que formen parte del contenido (p. ej. un artículo sobre la cocina española).
EN_PICTO = "a sequence of pictograms"
ES_PICTO = "una secuencia de pictogramas"
LANG_PATTERNS = [
    # inglés
    (re.compile(r"\b(?:the\s+)?Spanish(?:\s+language)?\b", re.I), EN_PICTO),
    # español: preposición + artículo contraído
    (re.compile(r"\bal\s+(?:idioma\s+)?(?:español|castellano)\b", re.I), "a " + ES_PICTO),
    (re.compile(r"\b(?:el\s+)?(?:idioma\s+)?(?:español|castellano)\b", re.I), ES_PICTO),
]


def _match_case(text: str, start: int, repl: str) -> str:
    # Mayúscula solo al inicio de línea u oración ("Español:" -> "Una secuencia...:");
    # "in Spanish" queda en minúscula aunque el gentilicio inglés se escriba con mayúscula.
    before = text[:start].rstrip(" \t")
    at_start = before == "" or before[-1] in "\n.!?¿¡:"
    return repl[0].upper() + repl[1:] if at_start else repl


def replace_target_language(text: str):
    """Devuelve (texto_modificado, se_reemplazó)."""
    best = None  # (start, end, repl) de la última mención en el texto
    for pat, repl in LANG_PATTERNS:
        for m in pat.finditer(text):
            # la mención que termina más tarde; a igual final, la más larga ("al español")
            if best is None or m.end() > best[1] or (m.end() == best[1] and m.start() < best[0]):
                best = (m.start(), m.end(), repl)
    if best is None:
        return text, False
    st, en, repl = best
    return text[:st] + _match_case(text, st, repl) + text[en:], True


def build_prompt(batch):
    out = [replace_target_language(t) for t in batch["inputs"]]
    return {"input": [o[0] for o in out], "lang_replaced": [o[1] for o in out]}


# --------------------------------------------------------------------------- #
# PASO 4 — target -> IDs de pictogramas
# --------------------------------------------------------------------------- #
_GRAMMAR = None


def get_grammar(args):
    global _GRAMMAR
    if _GRAMMAR is None:   # una instancia por proceso (num_proc)
        import spacy
        nlp = spacy.load(args.spacy_model, disable=["textcat"])
        lex = ArasaacLexicon.load(args.arasaac_cache, args.extra_lexicon)
        _GRAMMAR = Text2PictoES(lex, nlp, level=args.level)
    return _GRAMMAR


def to_pictos(batch, args):
    g = get_grammar(args)
    results = g.translate_batch(batch["target_es"])
    return {
        "picto_ids": [r.ids for r in results],
        "picto_tokens": [r.tokens for r in results],
        "target_picto": [" ".join(map(str, r.ids)) for r in results],
        "n_content": [r.n_content for r in results],
        "oov_terms": [r.oov for r in results],
        "oov_rate": [round(r.oov_rate, 4) for r in results],
    }


# --------------------------------------------------------------------------- #
def to_chat(ex):
    return {"messages": [{"role": "user", "content": ex["input"]},
                         {"role": "assistant", "content": ex["target_picto"]}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_files", default=DEFAULT_FILES)
    ap.add_argument("--max_records", type=int, default=None)
    ap.add_argument("--drop_unreplaced", action="store_true",
                    help="descarta registros cuyo input no menciona el idioma de salida")
    ap.add_argument("--level", type=int, choices=[1, 2, 3], default=3)
    ap.add_argument("--spacy_model", default="es_core_news_lg")
    ap.add_argument("--arasaac_cache", default="arasaac_es.json")
    ap.add_argument("--extra_lexicon", default=None, help="JSON {término: id} con prioridad")
    ap.add_argument("--min_words", type=int, default=3)
    ap.add_argument("--max_words", type=int, default=200)
    ap.add_argument("--lid_threshold", type=float, default=0.80)
    ap.add_argument("--max_oov_rate", type=float, default=0.25)
    ap.add_argument("--min_pictos", type=int, default=2)
    ap.add_argument("--num_proc", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="out")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Descargar el léxico una sola vez antes de lanzar procesos
    ArasaacLexicon.load(args.arasaac_cache, args.extra_lexicon)

    ds = load_xp3mt(args.data_files, args.max_records)                         # Paso 1
    print(f"[paso 1] {len(ds):,} registros cargados")
    ds = clean(ds, args)                                                       # Paso 2
    ds = ds.map(build_prompt, batched=True, num_proc=args.num_proc, desc="prompts")  # Paso 3
    n_rep = sum(ds["lang_replaced"])
    print(f"[paso 3] idioma de salida reemplazado en {n_rep:,}/{len(ds):,} inputs")
    if args.drop_unreplaced:
        ds = ds.filter(lambda b: b["lang_replaced"], batched=True, num_proc=args.num_proc)
    ds = ds.map(to_pictos, fn_kwargs={"args": args}, batched=True, batch_size=512,
                num_proc=args.num_proc, desc="texto->pictos")                  # Paso 4
    n_before = len(ds)
    ds = ds.filter(lambda b: [len(i) >= args.min_pictos and o <= args.max_oov_rate
                              for i, o in zip(b["picto_ids"], b["oov_rate"])],
                   batched=True, num_proc=args.num_proc, desc="calidad picto")
    print(f"[paso 4] {n_before:,} -> {len(ds):,} tras filtrar OOV>{args.max_oov_rate}")

    ds = ds.map(to_chat, num_proc=args.num_proc)
    ds = ds.remove_columns(["targets", "inputs"]).rename_column("target_es", "target")
    ds = ds.select_columns(["id", "input", "target", "target_picto", "messages", "picto_ids",
                            "picto_tokens", "n_content", "oov_terms", "oov_rate", "lang_replaced"])

    # Splits 90/5/5
    tmp = ds.train_test_split(test_size=0.10, seed=args.seed)
    vt = tmp["test"].train_test_split(test_size=0.50, seed=args.seed)
    dd = DatasetDict(train=tmp["train"], validation=vt["train"], test=vt["test"])
    for split, d in dd.items():
        d.to_json(os.path.join(args.out_dir, f"{split}.jsonl"), force_ascii=False)
    dd.save_to_disk(os.path.join(args.out_dir, "hf_dataset"))

    # Insights (Paso 5)
    from evaluate_picto import dataset_stats
    lex = ArasaacLexicon.load(args.arasaac_cache, args.extra_lexicon)
    stats = dataset_stats(dd["train"], lex)
    stats["cleaning_funnel"] = getattr(args, "_clean_counts", {})
    stats["splits"] = {k: len(v) for k, v in dd.items()}
    with open(os.path.join(args.out_dir, "dataset_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: stats[k] for k in ("n_examples", "vocab_coverage", "oov_rate_tokens",
                                             "seq_len")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
