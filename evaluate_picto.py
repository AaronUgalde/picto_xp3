"""
evaluate_picto.py
-----------------
Paso 5: métricas para secuencias de IDs de pictogramas + insights del dataset.

Métricas (todas sobre tokens = IDs, nunca sobre subpalabras del LLM):
    BLEU            sacrebleu, tokenize="none"   (como en Macaire et al., Tabla 7)
    TER             sacrebleu                    (Snover et al., 2006)
    PER (PictoER)   (S + D + I) / N_ref  a nivel corpus  (análogo al WER)
    ROUGE-L F1      LCS sobre IDs
    Levenshtein     distancia de edición media y normalizada por oración
    Exact Match     % de secuencias idénticas
    Picto P/R/F1    bolsa de pictos (recortada): ¿se eligieron los pictos correctos,
                    independientemente del orden?
    Validez         % de salidas parseables y % de IDs que existen en ARASAAC

Uso:
    python evaluate_picto.py --refs out/test.jsonl --preds preds.jsonl \
        [--arasaac_cache arasaac_es.json] [--show 10]
    preds.jsonl: {"id": ..., "prediction": "<texto generado por el modelo>"}
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from statistics import mean, median
from typing import Dict, List, Optional, Sequence

ID_RE = re.compile(r"\d+")


def parse_ids(text: str) -> List[str]:
    """Extrae los IDs de la salida libre del LLM ('6964 2317', '[6964, 2317]'...)."""
    return ID_RE.findall(text or "")


def levenshtein(a: Sequence, b: Sequence) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def lcs_len(a: Sequence, b: Sequence) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def compute_metrics(hyps: List[List[str]], refs: List[List[str]],
                    valid_ids: Optional[set] = None) -> Dict[str, float]:
    import sacrebleu
    assert len(hyps) == len(refs)
    hyp_s = [" ".join(h) for h in hyps]
    ref_s = [" ".join(r) for r in refs]

    bleu = sacrebleu.corpus_bleu(hyp_s, [ref_s], tokenize="none", force=True).score
    ter = sacrebleu.metrics.TER().corpus_score(hyp_s, [ref_s]).score

    edits = [levenshtein(h, r) for h, r in zip(hyps, refs)]
    n_ref = sum(len(r) for r in refs)
    per = 100 * sum(edits) / max(n_ref, 1)

    rl = []
    for h, r in zip(hyps, refs):
        l = lcs_len(h, r)
        p, rc = (l / len(h) if h else 0.0), (l / len(r) if r else 0.0)
        rl.append(2 * p * rc / (p + rc) if p + rc else 0.0)

    tp = fp = fn = 0
    for h, r in zip(hyps, refs):
        ch, cr = Counter(h), Counter(r)
        inter = sum((ch & cr).values())
        tp, fp, fn = tp + inter, fp + sum(ch.values()) - inter, fn + sum(cr.values()) - inter
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0

    out = {
        "BLEU": round(bleu, 2),
        "TER": round(ter, 2),
        "PER(%)": round(per, 2),
        "ROUGE-L_F1": round(100 * mean(rl), 2),
        "Levenshtein_mean": round(mean(edits), 3),
        "Levenshtein_norm(%)": round(100 * mean(e / max(len(r), 1) for e, r in zip(edits, refs)), 2),
        "ExactMatch(%)": round(100 * mean(h == r for h, r in zip(hyps, refs)), 2),
        "Picto_P(%)": round(100 * prec, 2),
        "Picto_R(%)": round(100 * rec, 2),
        "Picto_F1(%)": round(100 * (2 * prec * rec / (prec + rec) if prec + rec else 0), 2),
        "EmptyOutput(%)": round(100 * mean(len(h) == 0 for h in hyps), 2),
        "LenRatio(hyp/ref)": round(sum(map(len, hyps)) / max(n_ref, 1), 3),
    }
    if valid_ids is not None:
        all_h = [x for h in hyps for x in h]
        out["ValidID(%)"] = round(100 * mean(int(x) in valid_ids for x in all_h), 2) if all_h else 0.0
    return out


def metrics_by_length(hyps, refs, bins=((1, 3), (4, 6), (7, 10), (11, 15), (16, 10**9))):
    """Desglose por longitud de referencia (similar a la Tabla 8 del artículo)."""
    rows = {}
    for lo, hi in bins:
        idx = [i for i, r in enumerate(refs) if lo <= len(r) <= hi]
        if idx:
            m = compute_metrics([hyps[i] for i in idx], [refs[i] for i in idx])
            rows[f"{lo}-{hi if hi < 10**9 else '+'}"] = {
                "n": len(idx), "BLEU": m["BLEU"], "PER(%)": m["PER(%)"], "ExactMatch(%)": m["ExactMatch(%)"]}
    return rows


# --------------------------------------------------------------------------- #
# Insights del dataset transformado
# --------------------------------------------------------------------------- #
def _hist(values, edges):
    counts = Counter()
    for v in values:
        for lo, hi in edges:
            if lo <= v <= hi:
                counts[f"{lo}-{hi}"] += 1
                break
    return dict(counts)


def dataset_stats(ds, lexicon=None, top_k: int = 30) -> dict:
    ids = ds["picto_ids"]
    lens = [len(x) for x in ids]
    src_col = "target" if "target" in ds.column_names else "target_es"
    src_lens = [len(t.split()) for t in ds[src_col]]
    used = Counter(i for seq in ids for i in seq)
    oov = Counter(t for terms in ds["oov_terms"] for t in terms)
    n_content = sum(ds["n_content"])
    n_oov = sum(oov.values())
    stats = {
        "n_examples": len(ds),
        "seq_len": {"mean": round(mean(lens), 2), "median": median(lens),
                    "min": min(lens), "max": max(lens),
                    "hist": _hist(lens, [(1, 3), (4, 6), (7, 10), (11, 15), (16, 25), (26, 10**6)])},
        "src_words_mean": round(mean(src_lens), 2),
        "compression_ratio(pictos/palabras)": round(sum(lens) / max(sum(src_lens), 1), 3),
        "unique_pictos_used": len(used),
        "vocab_coverage": (round(len(used) / len(lexicon), 4) if lexicon else None),
        "oov_rate_tokens": round(n_oov / max(n_content, 1), 4),
        "examples_with_any_oov(%)": round(100 * mean(len(t) > 0 for t in ds["oov_terms"]), 2),
        "top_pictos": [(i, lexicon.label(i) if lexicon else "", c) for i, c in used.most_common(top_k)],
        "top_oov_lemmas": oov.most_common(top_k),
    }
    # Concentración: cuántos pictos cubren el 80 % de las ocurrencias
    total, acc, k80 = sum(used.values()), 0, 0
    for _, c in used.most_common():
        acc += c
        k80 += 1
        if acc >= 0.8 * total:
            break
    stats["pictos_for_80pct_mass"] = k80
    return stats


def comparison_table(rows: List[dict], lexicon=None, n: int = 10) -> str:
    """rows: dicts con source_text/target_es, ref_ids, pred_ids."""
    lab = (lambda i: lexicon.label(i)) if lexicon else (lambda i: "")
    out = []
    for r in rows[:n]:
        out.append("-" * 80)
        out.append(f"Texto origen : {r.get('source_text', r.get('target_es', ''))[:200]}")
        out.append(f"Target (es)  : {r.get('target_es', '')[:200]}")
        out.append("Referencia   : " + " ".join(f"{i}[{lab(i)}]" for i in r["ref_ids"]))
        out.append("Predicción   : " + " ".join(f"{i}[{lab(i)}]" for i in r["pred_ids"]))
        out.append(f"Levenshtein  : {levenshtein(r['pred_ids'], r['ref_ids'])}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--arasaac_cache", default=None)
    ap.add_argument("--show", type=int, default=10)
    ap.add_argument("--out", default="eval_report.json")
    args = ap.parse_args()

    refs = {}
    with open(args.refs, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            refs[ex["id"]] = ex
    preds = {}
    with open(args.preds, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            preds[p["id"]] = parse_ids(p["prediction"])
    keys = [k for k in refs if k in preds]
    print(f"{len(keys)} pares ref/pred ({len(refs) - len(keys)} refs sin predicción)")

    lex = None
    if args.arasaac_cache:
        from arasaac_lexicon import ArasaacLexicon
        lex = ArasaacLexicon.load(args.arasaac_cache)
    H = [preds[k] for k in keys]
    R = [[str(i) for i in refs[k]["picto_ids"]] for k in keys]
    report = {"overall": compute_metrics(H, R, set(lex.id2label) if lex else None),
              "by_ref_length": metrics_by_length(H, R)}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    rows = [{"target_es": refs[k].get("target", refs[k].get("target_es", "")), "ref_ids": R[i], "pred_ids": H[i]}
            for i, k in enumerate(keys)]
    print(comparison_table(rows, lex, args.show))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
