"""
visualize_pairs.py
------------------
Genera un HTML con pares (input, target, pictogramas) mostrando las imágenes
de ARASAAC. Se abre en el navegador (necesita internet para las imágenes).

Uso:
    python visualize_pairs.py --data out_dev/train.jsonl --n 30
    python visualize_pairs.py --data out_dev/train.jsonl --n 30 --max_len 15 --seed 7
"""
import argparse
import html
import json
import random
import webbrowser
from pathlib import Path

IMG = "https://static.arasaac.org/pictograms/{id}/{id}_300.png"

CSS = """
body{font-family:-apple-system,system-ui,sans-serif;margin:24px;background:#f6f6f4;color:#1d1d1f}
h1{font-size:20px}.meta{color:#666;font-size:13px;margin-bottom:20px}
.card{background:#fff;border:1px solid #e3e3e0;border-radius:10px;padding:16px;margin-bottom:18px}
.lbl{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:#888;margin-top:8px}
.input{font-size:13px;color:#444;white-space:pre-wrap;max-height:110px;overflow:auto;
       background:#fafafa;border-radius:6px;padding:8px}
.target{font-size:15px;margin:4px 0 6px}
.seq{display:flex;flex-wrap:wrap;gap:8px;margin-top:6px}
.p{width:92px;text-align:center;font-size:11px;color:#333}
.p img{width:84px;height:84px;object-fit:contain;border:1px solid #ddd;border-radius:6px;background:#fff}
.p.mk img{border:2px solid #d9822b}.id{color:#999;font-size:10px}
.oov{font-size:12px;color:#b3261e;margin-top:8px}
"""


def card(ex):
    e = html.escape
    pictos = []
    for pid, tok in zip(ex["picto_ids"], ex["picto_tokens"]):
        mk = " mk" if tok.startswith("<") else ""   # marcadores gramaticales en naranja
        pictos.append(f'<div class="p{mk}"><img loading="lazy" src="{IMG.format(id=pid)}">'
                      f'<div>{e(tok)}</div><div class="id">{pid}</div></div>')
    oov = ""
    if ex.get("oov_terms"):
        oov = f'<div class="oov">Sin picto ({ex["oov_rate"]:.0%}): {e(", ".join(ex["oov_terms"]))}</div>'
    return (f'<div class="card"><div class="lbl">input #{ex["id"]}</div>'
            f'<div class="input">{e(ex["input"])}</div>'
            f'<div class="lbl">target</div><div class="target">{e(ex["target"])}</div>'
            f'<div class="lbl">target_picto ({len(ex["picto_ids"])})</div>'
            f'<div class="seq">{"".join(pictos)}</div>{oov}</div>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out_dev/train.jsonl")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--max_len", type=int, default=None, help="solo secuencias de <= N pictos")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="pares_pictos.html")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    if args.max_len:
        rows = [r for r in rows if len(r["picto_ids"]) <= args.max_len]
    sample = random.Random(args.seed).sample(rows, min(args.n, len(rows)))

    page = (f'<!doctype html><meta charset="utf-8"><title>Pares texto→pictos</title>'
            f'<style>{CSS}</style><h1>Pares texto → pictogramas ARASAAC</h1>'
            f'<div class="meta">{len(sample)} de {len(rows)} ejemplos de {html.escape(args.data)} · '
            f'borde naranja = marcador gramatical (pasado, negación, imperativo…)</div>'
            + "".join(card(r) for r in sample))
    Path(args.out).write_text(page, encoding="utf-8")
    print(f"Escrito {args.out}")
    webbrowser.open(Path(args.out).resolve().as_uri())


if __name__ == "__main__":
    main()
