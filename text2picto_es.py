"""
text2picto_es.py
----------------
Gramática basada en reglas texto(es) -> secuencia de IDs ARASAAC.
Adaptación al español de las 10 reglas de Macaire et al. (2024, §3.2.2)
y del pipeline de la Fig. 3 (spaCy -> NER -> desambiguación -> mapeo léxico).

Niveles de traducción (§3.2.2):
    1: solo palabras léxicas (sustantivos, nombres propios, verbos, números)
    2: nivel 1 + pronombres (género/número)
    3: se intenta traducir todo término con pictograma (opción del artículo)

Reglas implementadas:
    R1  Marcador de tiempo   : picto PASADO/FUTURO ANTES del sujeto del grupo verbal
    R2  Imperativo           : picto '!' al FINAL de la oración
    R3  Verbo pronominal     : 'lavarse' si existe; si no, 'lavar'
    R4  Singular/Plural      : forma plural si existe; si no, lema
    R5  Números              : un picto por número si existe
    R6  Negación             : picto NO DESPUÉS del grupo verbal; 'no/nunca...' no se traducen
    R7  Prefijo              : des-/in-/im-/re- + palabra -> [picto prefijo, picto palabra]
    R8  Expresión multipalabra: búsqueda voraz hasta 8 palabras
    R9  Disfluencias         : se eliminan (eh, mmm...)
    R10 Entidades nombradas  : PER/LOC/ORG -> picto por defecto si no están en el léxico
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from arasaac_lexicon import ArasaacLexicon

# ID de "passé" usado en el artículo (Tabla 3). Los IDs de ARASAAC son
# independientes del idioma, así que sirve como valor por defecto.
DEFAULT_SPECIAL_KEYWORDS = {
    "PAST":   ["pasado", "ayer"],
    "FUTURE": ["futuro", "mañana"],
    "NEG":    ["no"],
    "IMP":    ["signo de exclamación", "exclamación", "!"],
    "AGAIN":  ["otra vez", "repetir"],
    "PER":    ["persona"],
    "LOC":    ["lugar", "ciudad"],
    "ORG":    ["empresa", "organización"],
}
DEFAULT_SPECIAL_IDS = {"PAST": 9839}

DISFLUENCIES = {"eh", "ehh", "ehm", "em", "emm", "mm", "mmm", "hmm", "hm", "hum",
                "uh", "um", "ah", "aah", "ajá", "eeh", "esteee"}
NEG_WORDS = {"no", "nunca", "jamás", "tampoco"}
REFLEXIVE = {"me", "te", "se", "nos", "os"}
# persona/número de cada clítico, para decidir si es reflexivo ("me lavo" sí, "me ves" no)
CLITIC_PN = {"me": ("1", "Sing"), "te": ("2", "Sing"), "nos": ("1", "Plur"), "os": ("2", "Plur")}
# marcadores discursivos eliminados solo cuando funcionan como muletilla
DISCOURSE = {"bueno", "pues", "vale", "este", "digamos", "osea"}
# auxiliares que solo aportan tiempo/aspecto/voz (los modales poder/deber/querer SÍ se traducen)
TENSE_AUX = {"haber", "ir", "estar", "ser"}
PREFIXES = {"des": "NEG", "in": "NEG", "im": "NEG", "re": "AGAIN"}
CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "INTJ", "NUM"}
LEVEL_POS = {
    1: {"NOUN", "PROPN", "VERB", "NUM"},
    2: {"NOUN", "PROPN", "VERB", "NUM", "PRON"},
    3: {"NOUN", "PROPN", "VERB", "NUM", "PRON", "ADJ", "ADV", "INTJ",
        "DET", "ADP", "CCONJ", "SCONJ", "AUX"},
}
# En nivel 3 las palabras gramaticales se traducen solo si existe picto,
# y su ausencia NO cuenta como pérdida (OOV).
FUNCTION_POS = {"DET", "ADP", "CCONJ", "SCONJ", "PRON", "AUX"}


@dataclass
class PictoResult:
    ids: List[int] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)   # 'picto token' (término) de cada ID
    n_content: int = 0                                 # términos léxicos que debían traducirse
    oov: List[str] = field(default_factory=list)       # lemas sin pictograma

    @property
    def oov_rate(self) -> float:
        return len(self.oov) / self.n_content if self.n_content else 0.0


class Text2PictoES:
    def __init__(self, lexicon: ArasaacLexicon, nlp, level: int = 3,
                 use_prefix_rule: bool = True,
                 special_ids: Optional[Dict[str, int]] = None):
        assert level in (1, 2, 3)
        self.lex, self.nlp, self.level = lexicon, nlp, level
        self.use_prefix_rule = use_prefix_rule
        self.special: Dict[str, Optional[int]] = {}
        for name, kws in DEFAULT_SPECIAL_KEYWORDS.items():
            pid = (special_ids or {}).get(name) or DEFAULT_SPECIAL_IDS.get(name) \
                  or lexicon.first_available(kws)
            self.special[name] = pid
            if pid is None:
                print(f"[grammar] AVISO: sin picto para marcador {name}; se omitirá")

    # ------------------------------------------------------------------ #
    def __call__(self, text: str) -> PictoResult:
        return self.translate_doc(self.nlp(text))

    def translate_batch(self, texts: List[str], batch_size: int = 256) -> List[PictoResult]:
        return [self.translate_doc(d) for d in self.nlp.pipe(texts, batch_size=batch_size)]

    # ------------------------------------------------------------------ #
    def _sp(self, name: str) -> Optional[Tuple[int, str]]:
        pid = self.special.get(name)
        return (pid, f"<{name}>") if pid is not None else None

    @staticmethod
    def _is_reflexive(t) -> bool:
        low = t.lower_
        if t.pos_ != "PRON" or low not in REFLEXIVE or t.head.pos_ not in ("VERB", "AUX"):
            return False
        if low == "se" or t.dep_.startswith("expl") or "Reflex=Yes" in str(t.morph):
            return True
        # me/te/nos/os: reflexivo si concuerda en persona y número con el verbo
        v = t.head
        per, num = v.morph.get("Person"), v.morph.get("Number")
        return bool(per) and (per[0], num[0] if num else None) == CLITIC_PN[low]

    @staticmethod
    def _is_disfluency(t) -> bool:
        if t.lower_ in DISFLUENCIES:
            return True
        nxt = t.nbor(1) if t.i + 1 < len(t.doc) else None
        return t.lower_ in DISCOURSE and (t.pos_ == "INTJ" or t.dep_ == "discourse"
                                          or (nxt is not None and nxt.text == ","))

    @staticmethod
    def _is_negation(t) -> bool:
        return t.lower_ in NEG_WORDS and t.dep_ in ("advmod", "neg", "dep") \
               and t.head.pos_ in ("VERB", "AUX", "ADJ")

    @staticmethod
    def _verb_group_end(v) -> int:
        """Último índice del grupo verbal: verbo + auxiliares + cadena de xcomp."""
        end = v.i
        stack = [v]
        while stack:
            cur = stack.pop()
            end = max(end, cur.i)
            for c in cur.children:
                if c.dep_ in ("aux", "aux:pass", "cop") or (c.dep_ == "xcomp" and c.pos_ in ("VERB", "AUX")):
                    stack.append(c)
        return end

    @staticmethod
    def _tense(v) -> Optional[str]:
        auxs = [c for c in v.children if c.dep_ in ("aux", "aux:pass")]
        morphs = [v.morph] + [a.morph for a in auxs]
        # perífrasis de futuro: ir a + infinitivo ("va a comer")
        if "Inf" in v.morph.get("VerbForm") and any(a.lemma_ == "ir" for a in auxs):
            return "FUTURE"
        # pretérito perfecto compuesto: haber + participio ("he comido")
        if "Part" in v.morph.get("VerbForm") and any(a.lemma_ == "haber" for a in auxs):
            return "PAST"
        for m in morphs:
            t = m.get("Tense")
            if "Past" in t or "Imp" in t:
                return "PAST"
            if "Fut" in t:
                return "FUTURE"
        return None

    def _lookup_token(self, t) -> Tuple[Optional[int], str]:
        pos = t.pos_
        lemma = t.lemma_.lower()
        # R5 números
        if pos == "NUM" or t.like_num:
            pid = self.lex.lookup(t.lower_) or self.lex.lookup(t.text)
            return pid, t.lower_
        # R3 verbo pronominal
        if pos in ("VERB", "AUX") and any(self._is_reflexive(c) for c in t.children):
            pid = self.lex.lookup(lemma + "se", pos="VERB")
            if pid is not None:
                return pid, lemma + "se"
        # Pronombres/determinantes: la forma de superficie conserva género y número
        # (spaCy lematiza "ella" -> "él", "nosotras" -> "yo"), que es justo lo que el nivel 2 quiere marcar
        if pos in ("PRON", "DET"):
            pid = self.lex.lookup(t.lower_, pos=pos)
            if pid is not None:
                return pid, t.lower_
        # R4 plural
        if pos == "NOUN" and "Plur" in t.morph.get("Number"):
            pid = self.lex.lookup(t.lower_, pos=pos, plural=True)
            if pid is not None:
                return pid, t.lower_
        pid = self.lex.lookup(lemma, pos=pos, plural=False if pos == "NOUN" else None)
        if pid is not None:
            return pid, lemma
        pid = self.lex.lookup(t.lower_, pos=pos)
        return pid, (lemma if pid is None else t.lower_)

    def _prefix_rule(self, t) -> Optional[List[Tuple[int, str]]]:
        low, lemma = t.lower_, t.lemma_.lower()
        for pre, marker in PREFIXES.items():
            for form in (lemma, low):
                rest = form[len(pre):]
                if form.startswith(pre) and len(rest) >= 4:
                    stem = self.lex.lookup(rest, pos=t.pos_)
                    sp = self._sp(marker)
                    if stem is not None and sp is not None:
                        return [sp, (stem, rest)]
        return None

    # ------------------------------------------------------------------ #
    def translate_doc(self, doc) -> PictoResult:
        res = PictoResult()
        n = len(doc)
        emit: Dict[int, List[Tuple[int, str]]] = {}
        pre: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
        post: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
        consumed = set()
        allowed = LEVEL_POS[self.level]

        # ---- R8: expresiones multipalabra (voraz, más larga primero) ----
        i = 0
        while i < n:
            hit = False
            # +1: la variante pronominal ('me lavo' -> 'lavarse') ocupa un token más en superficie
            for L in range(min(self.lex.max_ngram + 1, 8, n - i), 1, -1):
                span = doc[i:i + L]
                if any(t.is_punct or t.is_space for t in span):
                    continue
                if not any(t.pos_ in CONTENT_POS for t in span):
                    continue
                keys = [span.text, " ".join(t.lemma_ for t in span)]
                # variante pronominal: "me lavo los dientes" -> "lavarse los dientes"
                if L >= 2 and span[0].lower_ in REFLEXIVE and span[1].pos_ == "VERB":
                    keys.append(" ".join([span[1].lemma_ + "se"] + [t.text for t in span[2:]]))
                for k in keys:
                    if not self.lex.has_multiword(k):
                        continue
                    pid = self.lex.lookup(k)
                    if pid is not None:
                        emit[i] = [(pid, k.lower())]
                        consumed.update(range(i, i + L))
                        res.n_content += 1
                        hit = True
                        break
                if hit:
                    break
            i = i + L if hit else i + 1

        # ---- R10: entidades nombradas ----
        for ent in doc.ents:
            if ent.label_ not in ("PER", "LOC", "ORG") or any(t.i in consumed for t in ent):
                continue
            res.n_content += 1
            pid = self.lex.lookup(ent.text, pos="PROPN")
            if pid is not None:
                emit[ent.start] = [(pid, ent.text.lower())]
            else:
                sp = self._sp(ent.label_)
                if sp is not None:
                    emit[ent.start] = [sp]
                else:
                    res.oov.append(ent.text)
            consumed.update(range(ent.start, ent.end))

        # ---- marcadores gramaticales (R1, R2, R6) ----
        for t in doc:
            # R6 negación: tras el grupo verbal
            if self._is_negation(t):
                consumed.add(t.i)
                head = t.head
                sp = self._sp("NEG")
                if sp is not None:
                    post[self._verb_group_end(head)].append(sp)
                continue
            # R1 tiempo: antes del sujeto (solo verbos con tiempo propio)
            is_verb = t.pos_ in ("VERB", "AUX") and t.dep_ not in ("aux", "aux:pass", "cop", "xcomp")
            is_cop = t.dep_ == "cop"          # "fue un terremoto": el tiempo vive en la cópula
            if is_verb or is_cop:
                tense = self._tense(t)
                sp = self._sp(tense) if tense else None
                if sp is not None:
                    pred = t.head if is_cop else t
                    subj = [c for c in pred.children if c.dep_.startswith("nsubj")
                            and not self._is_disfluency(c)]
                    left = [t.i, pred.i] + [c.i for c in pred.children
                                            if c.dep_ in ("aux", "aux:pass", "cop")]
                    if subj:
                        left.append(subj[0].left_edge.i)
                    pre[min(left)].append(sp)
        # R2 imperativo: oración cuya raíz está en modo imperativo
        for sent in doc.sents:
            r = sent.root
            if r.pos_ in ("VERB", "AUX") and "Imp" in r.morph.get("Mood"):
                sp = self._sp("IMP")
                if sp is not None:
                    post[sent.end - 1].append(sp)

        # ---- palabras sueltas ----
        for t in doc:
            if t.i in consumed or t.is_punct or t.is_space:
                continue
            if self._is_disfluency(t):                           # R9
                continue
            if self._is_reflexive(t):                            # absorbido por R3
                continue
            pos = t.pos_
            if pos == "AUX" and t.dep_ != "cop" and t.lemma_.lower() in TENSE_AUX:
                continue                                         # tiempo ya marcado por R1
            if pos == "PROPN" and pos not in allowed:
                continue
            if pos not in allowed:
                continue
            is_function = pos in FUNCTION_POS
            if not is_function:
                res.n_content += 1
            pid, term = self._lookup_token(t)
            if pid is not None:
                emit[t.i] = [(pid, term)]
                continue
            if self.use_prefix_rule and pos in ("ADJ", "NOUN", "VERB"):  # R7
                pr = self._prefix_rule(t)
                if pr:
                    emit[t.i] = pr
                    continue
            if not is_function:
                res.oov.append(t.lemma_.lower())

        # ---- ensamblado en orden lineal ----
        seq: List[Tuple[int, str]] = []
        for k in range(n):
            seq.extend(pre.get(k, []))
            seq.extend(emit.get(k, []))
            seq.extend(post.get(k, []))
        # colapsar marcadores idénticos consecutivos (p. ej. dos <PAST>)
        out: List[Tuple[int, str]] = []
        for item in seq:
            if out and item == out[-1] and item[1].startswith("<"):
                continue
            out.append(item)
        res.ids = [p for p, _ in out]
        res.tokens = [tok for _, tok in out]
        return res
