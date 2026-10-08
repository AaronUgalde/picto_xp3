# Metodología

Construcción de un corpus de instrucciones con salida en pictogramas ARASAAC a partir de `bigscience/xP3mt` (subconjunto español), para el fine-tuning de un LLM que genere secuencias de identificadores de pictogramas.

La conversión texto → pictogramas adapta al español la gramática basada en reglas de:

> Macaire, C., Dion, C., Arrigo, J., Lemaire, C., Esperança-Rodier, E., Lecouteux, B., & Schwab, D. (2024). *A Multimodal French Corpus of Aligned Speech, Text, and Pictogram Sequences for Speech-to-Pictogram Machine Translation*. LREC-COLING 2024, pp. 839–849.

En adelante se cita como "el artículo", con referencias a sus secciones (§) y tablas.

---

## 0. Visión general

```
xP3mt/es/merged_es.jsonl
        │  Paso 1  Ingesta (inputs, targets)
        ▼
  Paso 2  Limpieza: normalización → heurísticas → ID de idioma (fastText) → dedup
        ▼
  Paso 3  Reformulación del input: "Spanish/español" → "a sequence of pictograms / una secuencia de pictogramas"
        ▼
  Paso 4  target (es) → gramática de reglas (spaCy + léxico ARASAAC) → secuencia de IDs
        │           filtro de calidad (tasa OOV, longitud mínima)
        ▼
  Splits 90/5/5 → JSONL (formato chat) + DatasetDict + dataset_stats.json
        ▼
  Paso 5  Evaluación (BLEU, TER, PER, ROUGE-L, Levenshtein, EM, Picto-F1) e inspección visual
```

| Módulo | Paso |
|---|---|
| `build_dataset.py` | 1, 2, 3, orquestación de 4, exportación |
| `arasaac_lexicon.py` | 4 (léxico) |
| `text2picto_es.py` | 4 (gramática) |
| `evaluate_picto.py` | 5 (métricas y estadísticas) |
| `visualize_pairs.py` | 5 (inspección cualitativa) |

---

## 1. Paso 1: ingesta

- **Fuente:** `hf://datasets/bigscience/xP3mt/es/merged_es.jsonl`, unos 3.37 M registros y 7.8 GB. Se lee con el builder `json` de `datasets`, no con el script del repositorio, por compatibilidad con `datasets>=3`.
- **Campos:** `inputs` (prompt de la tarea) y `targets` (respuesta).
- **Precaución:** la carpeta `es/` también contiene los ~400 jsonl individuales por tarea que forman el merged. Cargar `es/*.jsonl` duplicaría el corpus, así que se usa solo el merged.
- **Muestreo:** con `--max_records N` se leen los primeros N registros en streaming, para desarrollo.

Contenido de la carpeta `es/`: tareas cuyo idioma objetivo es el español, más traducción en ambas direcciones (FLORES, Tatoeba), resúmenes de wiki_lingua y xlsum, y plantillas traducidas automáticamente (`*_esmt`). Por eso **no se puede asumir que el target esté en español**; eso lo resuelve el paso 2.

## 2. Paso 2: limpieza y filtrado

Se aplica sobre el `target`, en este orden:

1. **Normalización:** NFC, espacios colapsados, eliminación de espacios de ancho cero y comillas envolventes.
2. **Heurísticas** (`heuristic_ok`):
   - longitud entre `--min_words` (3) y `--max_words` (200) palabras;
   - sin patrones de código, HTML, SQL, URLs ni LaTeX;
   - al menos 70 % de caracteres alfabéticos (descarta números, símbolos y tablas);
   - sin repeticiones anómalas (`aaaaa`, `!!!!!`);
   - menos del 50 % de palabras en mayúsculas.
3. **Identificación de idioma:** modelo fastText `facebook/fasttext-language-identification` (NLLB-LID, 218 idiomas). Se conserva el registro si `P(spa_Latn) ≥ --lid_threshold` (0.80). Este filtro corre en un solo proceso, porque el modelo pesa ~1.2 GB y replicarlo en subprocesos agota la memoria (en macOS los subprocesos arrancan con *spawn*). Si fastText no está instalado, hay una heurística de respaldo más débil (diacríticos y palabras vacías).
4. **Deduplicación** de pares `(inputs, target)` idénticos (hash MD5).

## 3. Paso 3: reformulación del input

**Principio:** la tarea original de xP3mt se conserva **sin añadir instrucciones**. Solo se sustituye la mención del idioma de salida por la nueva modalidad de salida.

| Original | Reformulado |
|---|---|
| `Write a summary of the previous text in Spanish:` | `Write a summary of the previous text in a sequence of pictograms:` |
| `A text in Xhosa: … The same text in Spanish:` | `A text in Xhosa: … The same text in a sequence of pictograms:` |
| `Resume el texto en español:` | `Resume el texto en una secuencia de pictogramas:` |
| `Traduce al español:` | `Traduce a una secuencia de pictogramas:` |

Reglas de `replace_target_language`:

- **Patrones:** `(the) Spanish (language)` → `a sequence of pictograms`; `al (idioma) español/castellano` → `a una secuencia de pictogramas`; `(el) (idioma) español/castellano` → `una secuencia de pictogramas`.
- **Solo se reemplaza la mención que termina más tarde.** En las plantillas de xP3 la instrucción va al final, así no se alteran menciones que forman parte del contenido (*"The Spanish chef… Summarize in Spanish:"*). A igual posición final, gana la coincidencia más larga.
- **Mayúscula inicial** solo si la mención abre línea u oración (`Español:` → `Una secuencia de pictogramas:`).
- **Inputs sin mención** del idioma se conservan y quedan marcados con `lang_replaced=false`; `--drop_unreplaced` los descarta.

## 4. Paso 4: conversión del target a IDs ARASAAC

### 4.1 Léxico (`arasaac_lexicon.py`; artículo §3.2.1)

- **Fuente:** `GET https://api.arasaac.org/v1/pictograms/all/es`, con caché local en `arasaac_es.json`. Contiene unos 13.8 k pictogramas y 20.4 k términos.
- **Índice:** cada `keyword` y su `plural` apuntan a candidatos `(id, type, aac, es_plural, marcado_sex/violence)`. Las keywords de varias palabras permiten expresiones de hasta 8 palabras.
- **Entradas propias:** `--extra_lexicon` recibe un JSON `{término: id}` con prioridad absoluta. Sirve para curación manual o para reutilizar el léxico de ProPicto re-etiquetado al español.
- **Desambiguación por defecto,** que sustituye al WSD neuronal del artículo (Vial et al., 2019). Ante varios candidatos se ordenan por:
  1. no marcados como sexuales o violentos;
  2. `type` de la keyword compatible con la etiqueta UPOS de spaCy (1 nombre propio, 2 nombre común, 3 verbo, 4 descriptivo, 5 social, 6 misceláneo);
  3. plural concordante;
  4. `aac = true`;
  5. ID menor (picto más antiguo y canónico).

### 4.2 Análisis lingüístico

spaCy `es_core_news_lg` aporta tokenización, lematización, POS, rasgos morfológicos, dependencias y NER, como en la Fig. 3 del artículo.

### 4.3 Niveles de traducción (artículo §3.2.2)

| Nivel | Se traducen |
|---|---|
| 1 | sustantivos, nombres propios, verbos, números |
| 2 | nivel 1 + pronombres (género y número) |
| 3 (por defecto, como el corpus del artículo) | todo término con picto; las palabras de función sin picto se omiten sin contar como pérdida |

### 4.4 Reglas (adaptación de las 10 del artículo)

| # | Regla | Implementación en español |
|---|---|---|
| R1 | Marcador de tiempo | `<PAST>` o `<FUTURE>` antes del sujeto del grupo verbal. Pasado: `Tense=Past/Imp`, o *haber* + participio. Futuro: `Tense=Fut`, o *ir a* + infinitivo. En oraciones copulativas el tiempo se lee de la cópula. |
| R2 | Imperativo | `<IMP>` al final de una oración cuya raíz tiene `Mood=Imp`. |
| R3 | Verbo pronominal | Si hay clítico reflexivo, se busca *lema + se* (*lavarse*); si no existe, se usa el verbo solo. *me/te/nos/os* cuentan como reflexivos solo si concuerdan en persona y número con el verbo. |
| R4 | Singular/plural | Primero se busca la forma plural en el índice de plurales; si no está, el lema. |
| R5 | Números | Un picto por número, si existe. |
| R6 | Negación | *no, nunca, jamás, tampoco* no se traducen; se añade `<NEG>` después del grupo verbal (verbo + auxiliares + cadena de `xcomp`). |
| R7 | Prefijo | Si un término es OOV y empieza por *des-, in-, im-* (`<NEG>`) o *re-* (`<AGAIN>`), y la raíz de ≥ 4 letras está en el léxico, se emite [picto del prefijo, picto de la raíz]. |
| R8 | Expresión multipalabra | Búsqueda voraz, de la ventana más larga a la más corta (hasta 8 tokens), sobre la forma superficial, los lemas y una variante pronominal (*me lavo los dientes* → *lavarse los dientes*). Se exige al menos un token de contenido. |
| R9 | Disfluencias | Se eliminan *eh, mmm, hum…* y los marcadores discursivos (*bueno, pues, este*) cuando funcionan como muletilla (INTJ, `discourse` o seguidos de coma). |
| R10 | Entidades nombradas | `PER/LOC/ORG` se buscan primero en el léxico; si no están, reciben el picto por defecto `<PER>` (*persona*), `<LOC>` (*lugar*) o `<ORG>` (*empresa*). |

Otras decisiones:

- **Auxiliares:** se omiten los que solo aportan tiempo, aspecto o voz (*haber, ir, estar, ser* no copulativo); los modales (*poder, deber, querer*) sí se traducen.
- **Pronombres y determinantes** se buscan por su forma superficial antes que por el lema, para conservar el género (spaCy lematiza *ella* como *él*).
- **Marcadores especiales:** `<PAST>` usa el ID 9839, el mismo del artículo (Tabla 3); los IDs de ARASAAC no dependen del idioma. El resto se resuelve por palabra clave al cargar el léxico y se puede fijar con `special_ids`.
- **Ensamblado:** orden lineal de la oración; los marcadores se insertan antes o después según la regla, y se colapsan los marcadores idénticos consecutivos.

### 4.5 Filtro de calidad

Por cada target se registran `n_content` (términos léxicos a traducir) y `oov_terms` (lemas sin picto). Se descarta el registro si:

- la tasa de OOV supera `--max_oov_rate` (0.25), o
- la secuencia tiene menos de `--min_pictos` (2) pictos.

## 5. Salida

- **Splits:** 90/5/5 aleatorios con semilla 42 (`train/validation/test.jsonl`), más `hf_dataset/` en formato `DatasetDict`.
- **Campos por registro:** `id, input, target, target_picto, messages, picto_ids, picto_tokens, n_content, oov_terms, oov_rate, lang_replaced`.
  - `messages` = `[{"role":"user","content":input}, {"role":"assistant","content":target_picto}]`, listo para `trl.SFTTrainer`.
  - `target_picto` = IDs separados por espacios.

## 6. Paso 5: evaluación

`evaluate_picto.py` calcula todas las métricas sobre IDs completos, nunca sobre subpalabras del tokenizador:

| Métrica | Definición |
|---|---|
| BLEU | sacreBLEU con `tokenize="none"` (comparable con la Tabla 7 del artículo) |
| TER | sacreBLEU TER |
| PER (Picto Error Rate) | (S + D + I) / N_ref, a nivel corpus |
| ROUGE-L | F1 de la subsecuencia común más larga |
| Levenshtein | media y normalizada por la longitud de la referencia |
| Exact Match | % de secuencias idénticas |
| Picto P/R/F1 | bolsa de pictos con conteos recortados; mide la elección de pictos independientemente del orden |
| ValidID, EmptyOutput, LenRatio | fallos de formato propios de un LLM |

También incluye un desglose por longitud de la referencia (análogo a la Tabla 8 del artículo) y una tabla comparativa texto / referencia / predicción.

`dataset_stats.json` resume el corpus:

- embudo de limpieza;
- histograma de longitudes y ratio pictos/palabra;
- cobertura del vocabulario ARASAAC;
- tasa de OOV y lemas OOV más frecuentes;
- pictos más frecuentes y cuántos concentran el 80 % de las ocurrencias.

`visualize_pairs.py` genera un HTML con las imágenes de ARASAAC para inspección cualitativa.

## 7. Limitaciones conocidas

Ver [`RESULTADOS_out_dev.md`](RESULTADOS_out_dev.md) para la evidencia cuantitativa.

1. Las referencias provienen de una gramática, no de anotación humana; el techo del modelo es la calidad de la gramática.
2. El NER de spaCy confunde imperativos al inicio de oración con entidades.
3. Los imperativos irregulares o con enclíticos no se lematizan.
4. La regla de prefijo *re-* produce falsos positivos.
5. La desambiguación es heurística (POS + metadatos), sin contexto semántico.
6. La cobertura está limitada por el vocabulario de ARASAAC (OOV ≈ 18 %).

## 8. Reproducibilidad

- Semilla fija (`--seed 42`) para los splits; el resto del pipeline es determinista dado el modelo de spaCy y la versión del léxico.
- El léxico ARASAAC cambia con el tiempo. Para un experimento final, guarda la fecha de descarga de `arasaac_es.json` (o su hash) junto con los resultados.
- Registrar la versión de `datasets`, `spacy` y del modelo `es_core_news_*` usados.
