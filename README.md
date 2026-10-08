# picto_xp3: corpus de instrucciones texto → pictogramas ARASAAC

Pipeline que transforma el subconjunto español de [`bigscience/xP3mt`](https://huggingface.co/datasets/bigscience/xP3mt) en un corpus de instrucciones cuya salida es una **secuencia de identificadores de pictogramas [ARASAAC](https://arasaac.org)**. El corpus está pensado para hacer fine-tuning de LLMs en la tarea Text-to-Picto.

La conversión de texto a pictogramas adapta al español la gramática de reglas de Macaire et al. (2024), *A Multimodal French Corpus of Aligned Speech, Text, and Pictogram Sequences for Speech-to-Pictogram Machine Translation* (LREC-COLING).

Forma parte del Trabajo Terminal **TT 2027-A036, "Traducción Automática de Texto a Pictogramas mediante Modelos Transformer"** (ESCOM-IPN).

## Ejemplo

| input | target | target_picto |
|---|---|---|
| `…Pumice stones are available at most health and beauty stores…`<br>`===`<br>`Write a summary of the previous text in a sequence of pictograms:` | *Si quieres una solución sencilla, utiliza una lima esmeril para personas…* | `id1 id2 id3 …` |
| `A text in Xhosa: Ingenziwa icekeceke…`<br>`The same text in a sequence of pictograms:` | *Puede hacerse muy delgado y adherirse a otro metal…* | `id1 id2 id3 …` |

La tarea original de xP3mt se conserva; solo se cambia el idioma de salida ("Spanish" → "a sequence of pictograms").

## Documentación

- **[docs/METODOLOGIA.md](docs/METODOLOGIA.md):** metodología completa. Explica los pasos 1–5, el léxico, los niveles, las 10 reglas, el formato de salida, las métricas y las limitaciones.
- **[docs/RESULTADOS_out_dev.md](docs/RESULTADOS_out_dev.md):** resultados de la línea base (50 k registros), el análisis de errores y la hoja de ruta de mejoras.

## Estructura

```
picto_xp3/
├── build_dataset.py      # Pasos 1–4: ingesta, limpieza, reformulación, conversión, splits, exportación
├── arasaac_lexicon.py    # Léxico término → ID ARASAAC y desambiguación
├── text2picto_es.py      # Gramática de reglas español → pictos (niveles 1–3, reglas R1–R10)
├── evaluate_picto.py     # Paso 5: métricas (BLEU, TER, PER, ROUGE-L, Levenshtein, EM, Picto-F1) y estadísticas
├── visualize_pairs.py    # HTML con las imágenes ARASAAC de pares muestreados
├── docs/                 # Metodología y resultados
└── requirements.txt
```

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download es_core_news_lg
```

Si `fasttext` falla con NumPy 2, instala `fasttext-numpy2-wheel`.

## Uso

```bash
# Desarrollo: 50k registros en streaming
python build_dataset.py --max_records 50000 --out_dir out_dev --num_proc 4

# Corpus completo (~3.37 M registros, ~7.8 GB de descarga)
python build_dataset.py --num_proc 8 --out_dir out_full

# Inspección visual de pares
python visualize_pairs.py --data out_dev/train.jsonl --n 30 --max_len 25

# Evaluación de las predicciones de un modelo
#   preds.jsonl: {"id": ..., "prediction": "<texto generado>"}
python evaluate_picto.py --refs out_dev/test.jsonl --preds preds.jsonl --arasaac_cache arasaac_es.json
```

La primera ejecución descarga el léxico de ARASAAC (`arasaac_es.json`) y el modelo fastText de identificación de idioma.

### Opciones principales de `build_dataset.py`

| Opción | Por defecto | Descripción |
|---|---|---|
| `--max_records` | todo | Lee solo los primeros N registros (streaming) |
| `--level` | 3 | Nivel de traducción: 1 léxico, 2 + pronombres, 3 todo |
| `--spacy_model` | `es_core_news_lg` | Modelo de spaCy |
| `--min_words` / `--max_words` | 3 / 200 | Longitud del target |
| `--lid_threshold` | 0.80 | Probabilidad mínima de español (fastText) |
| `--max_oov_rate` | 0.25 | Fracción máxima de términos sin picto |
| `--min_pictos` | 2 | Longitud mínima de la secuencia |
| `--drop_unreplaced` | no | Descarta inputs sin mención del idioma de salida |
| `--extra_lexicon` | — | JSON `{término: id}` con prioridad sobre ARASAAC |
| `--num_proc` | 4 | Procesos para los pasos paralelizables |

## Formato de salida

`out_*/{train,validation,test}.jsonl`, `out_*/hf_dataset/` (`DatasetDict`) y `out_*/dataset_stats.json`.

Esquema de cada registro (valores ilustrativos):

```json
{
  "id": 22,
  "input": "... Write a summary of the previous text in a sequence of pictograms:",
  "target": "Si quieres una solución sencilla, utiliza una lima esmeril ...",
  "target_picto": "id1 id2 id3 ...",
  "messages": [{"role": "user", "content": "<input>"}, {"role": "assistant", "content": "<target_picto>"}],
  "picto_ids": [id1, id2, id3],
  "picto_tokens": ["término1", "término2", "término3"],
  "n_content": 31, "oov_terms": ["esmeril"], "oov_rate": 0.03, "lang_replaced": true
}
```

`messages` funciona directamente con `trl.SFTTrainer`.

## Licencias de los datos

- **xP3mt:** Apache 2.0.
- **Pictogramas ARASAAC:** autor Sergio Palao, origen ARASAAC (https://arasaac.org), licencia CC BY-NC-SA 4.0, propiedad del Gobierno de Aragón (España). Cualquier corpus o visualización que use sus pictogramas debe conservar esta atribución y su carácter no comercial. Por eso la caché `arasaac_es.json` y los corpus generados no se versionan en este repositorio.

## Referencias

- Macaire, C. et al. (2024). *A Multimodal French Corpus of Aligned Speech, Text, and Pictogram Sequences for Speech-to-Pictogram Machine Translation*. LREC-COLING 2024, 839–849.
- Muennighoff, N. et al. (2023). *Crosslingual Generalization through Multitask Finetuning*. ACL 2023 (xP3 / xP3mt).
- Vial, L., Lecouteux, B., & Schwab, D. (2019). *Sense Vocabulary Compression through the Semantic Knowledge of WordNet for Neural Word Sense Disambiguation*. GWC 2019.
