"""
arasaac_lexicon.py
------------------
Léxico restringido término -> ID de pictograma ARASAAC (equivalente al
"lexicon" de Macaire et al., 2024, §3.2.1, pero en español).

Fuente: API pública de ARASAAC
    GET https://api.arasaac.org/v1/pictograms/all/es
Cada pictograma trae: _id, keywords[{keyword, plural, type, meaning}],
aac, schematic, sex, violence, tags, categories...

`type` de la keyword (convención ARASAAC):
    1 = nombre propio, 2 = nombre común, 3 = verbo,
    4 = descriptivo (adj/adv), 5 = contenido social/expresión, 6 = misceláneo

Ese `type` se usa como desambiguación barata por categoría gramatical
(sustituye al módulo WSD neuronal del artículo; ver README para la mejora).
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

ARASAAC_ALL_URL = "https://api.arasaac.org/v1/pictograms/all/es"

# Categoría UD de spaCy -> type de keyword ARASAAC
POS2TYPE = {
    "PROPN": 1, "NOUN": 2, "VERB": 3, "AUX": 3,
    "ADJ": 4, "ADV": 4, "INTJ": 5,
    "PRON": 6, "DET": 6, "ADP": 6, "CCONJ": 6, "SCONJ": 6, "NUM": 6,
}


def normalize(term: str) -> str:
    term = unicodedata.normalize("NFC", term).lower().strip()
    term = re.sub(r"\s+", " ", term)
    return term


def strip_accents(term: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", term)
                   if unicodedata.category(c) != "Mn")


@dataclass(frozen=True)
class Candidate:
    pid: int
    kw_type: int
    aac: bool
    is_plural: bool
    flagged: bool  # sex / violence


class ArasaacLexicon:
    def __init__(self, pictos: Iterable[dict], extra: Optional[Dict[str, int]] = None):
        self.index: Dict[str, List[Candidate]] = {}
        self.index_noacc: Dict[str, List[Candidate]] = {}
        self.id2label: Dict[int, str] = {}
        self.max_ngram = 1

        for p in pictos:
            pid = int(p["_id"])
            flagged = bool(p.get("sex")) or bool(p.get("violence"))
            aac = bool(p.get("aac"))
            for kw in p.get("keywords", []):
                k = kw.get("keyword")
                if not k:
                    continue
                t = int(kw.get("type") or 6)
                self.id2label.setdefault(pid, k)
                self._add(k, Candidate(pid, t, aac, False, flagged))
                if kw.get("plural"):
                    self._add(kw["plural"], Candidate(pid, t, aac, True, flagged))

        # Entradas propias (p. ej. léxico ProPicto re-etiquetado al español,
        # o correcciones manuales). Tienen prioridad absoluta.
        self.overrides: Dict[str, int] = {normalize(k): int(v) for k, v in (extra or {}).items()}
        for k in self.overrides:
            self.max_ngram = max(self.max_ngram, len(k.split()))
        self.max_ngram = min(self.max_ngram, 8)  # el artículo mira hasta 8 palabras

    def _add(self, term: str, cand: Candidate):
        key = normalize(term)
        self.index.setdefault(key, []).append(cand)
        self.index_noacc.setdefault(strip_accents(key), []).append(cand)
        self.max_ngram = max(self.max_ngram, len(key.split()))

    # ------------------------------------------------------------------ #
    @staticmethod
    def _rank(cands: List[Candidate], pos: Optional[str], want_plural: Optional[bool]) -> int:
        want_type = POS2TYPE.get(pos) if pos else None

        def score(c: Candidate):
            return (
                not c.flagged,                                # evitar sex/violence
                want_type is not None and c.kw_type == want_type,
                want_plural is not None and c.is_plural == want_plural,
                c.aac,                                       # pictos marcados para CAA
                -c.pid,                                      # ID más bajo = picto más canónico
            )
        return max(cands, key=score).pid

    def lookup(self, term: str, pos: Optional[str] = None,
               plural: Optional[bool] = None, accent_fallback: bool = True) -> Optional[int]:
        key = normalize(term)
        if not key:
            return None
        if key in self.overrides:
            return self.overrides[key]
        cands = self.index.get(key)
        if not cands and accent_fallback:
            cands = self.index_noacc.get(strip_accents(key))
        if not cands:
            return None
        if plural is not None and pos is not None:
            # 'casas' solo debe resolver a candidatos cuya forma plural coincida
            exact = [c for c in cands if c.is_plural == plural]
            cands = exact or cands
        return self._rank(cands, pos, plural)

    def has_multiword(self, term: str) -> bool:
        key = normalize(term)
        return " " in key and (key in self.overrides or key in self.index)

    def first_available(self, terms: List[str]) -> Optional[int]:
        for t in terms:
            pid = self.lookup(t)
            if pid is not None:
                return pid
        return None

    def label(self, pid) -> str:
        try:
            return self.id2label.get(int(pid), str(pid))
        except (TypeError, ValueError):
            return str(pid)

    def __len__(self):
        return len(self.id2label)

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, cache_path: str = "arasaac_es.json",
             extra_lexicon: Optional[str] = None) -> "ArasaacLexicon":
        if not os.path.exists(cache_path):
            import requests
            print(f"[lexicon] Descargando ARASAAC (es) -> {cache_path}")
            r = requests.get(ARASAAC_ALL_URL, timeout=300)
            r.raise_for_status()
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(r.text)
        with open(cache_path, encoding="utf-8") as f:
            pictos = json.load(f)
        extra = None
        if extra_lexicon:
            with open(extra_lexicon, encoding="utf-8") as f:
                extra = json.load(f)  # {"término": id, ...}
        lex = cls(pictos, extra)
        print(f"[lexicon] {len(lex)} pictogramas, {len(lex.index)} términos, "
              f"n-grama máx = {lex.max_ngram}")
        return lex
