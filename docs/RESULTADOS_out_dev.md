# Resultados de la línea base: `out_dev` (50 k registros)

**Corrida:**

```bash
python build_dataset.py --max_records 50000 --out_dir out_dev --num_proc 4
```

**Configuración:** spaCy `es_core_news_lg`, nivel 3, `--max_oov_rate 0.25`, `--max_words 200`, semilla 42. Fecha: 2026-10-07.

Esta es la versión **v0 de la gramática**. Los números sirven de punto de comparación para las mejoras de la sección 4.

## 1. Embudo de datos

| Etapa | Registros | % del total |
|---|---:|---:|
| Leídos | 50,000 | 100 % |
| Tras heurísticas | 38,292 | 76.6 % |
| Tras ID de idioma (es) | 34,087 | 68.2 % |
| Tras dedup | 34,087 | 68.2 % |
| Tras filtro de calidad picto (OOV ≤ 25 %) | 8,685 | 17.4 % |
| train / validation / test | 7,816 / 434 / 435 | |

El filtro de OOV descarta ~75 % de los registros que sobreviven a la limpieza. Es la principal pérdida de datos.

## 2. Estadísticas del corpus resultante (train)

| Indicador | Valor |
|---|---|
| Longitud de secuencia (media / mediana / máx.) | 25.5 / 22 / 174 pictos |
| Ratio pictos/palabra | 0.87 |
| Pictos únicos usados | 4,070 (29.4 % del vocabulario ARASAAC) |
| Pictos que concentran el 80 % de las ocurrencias | 373 |
| Tasa OOV (tokens de contenido) | 17.96 % |
| Ejemplos con al menos un OOV | 80.5 % |
| Proporción de palabras de función en la salida | 38 % (*la, de, un, los, en, a…*) |

Marcadores emitidos sobre 199,333 pictos:

| Marcador | Ocurrencias |
|---|---:|
| `<LOC>` | 4,142 |
| `<PER>` | 3,564 |
| `<PAST>` | 3,190 |
| `<ORG>` | 1,381 |
| `<NEG>` | 1,117 |
| `<AGAIN>` | 581 |
| `<IMP>` | 512 |
| `<FUTURE>` | 459 |

## 3. Análisis de errores

### 3.1 Errores del sistema, que son corregibles

| # | Error | Evidencia | Ejemplo |
|---|---|---|---|
| E1 | Imperativos etiquetados como entidades por el NER | En 1,500 oraciones, 818 de 2,082 entidades PER/LOC/ORG (39 %) empiezan con un verbo o una palabra no-PROPN al inicio de oración. *lugar* es el 7.º picto más frecuente. | *Usa un estadiómetro* → `<LOC> un …` |
| E2 | Imperativos irregulares o con enclítico no lematizados | Top OOV: *haz* (309), *pon* (122), *asegúratir* (90), *mantén* (75), *tir* (55); *hacer*, *poner* y *mantener* sí existen en ARASAAC. Solo 512 `<IMP>` en un corpus dominado por instrucciones. | *Cepíllate los dientes. Haz gárgaras* → `<PAST> los dientes gárgaras` |
| E3 | Falsos positivos del prefijo *re-* | De 581 `<AGAIN>`, casi todos son erróneos: re+*tirar* (113), re+*únir* (89), re+*mojar* (73), re+*lleno*, re+*curso*, re+*acción*, re+*medio*… | *retirar* → `<AGAIN> tirar` |
| E3b | Falsos positivos de *des-/in-* (menos frecuentes) | *despedir* → `<NEG> pedir`, *desplegar* → `<NEG> plegar` (las formas correctas incluyen *descongelar* y *deshidratación*) | |
| E4 | La búsqueda sin acentos produce falsos amigos | | *Tómate un tiempo* → 🍅 *tomate* |
| E5 | Ruido de la fuente | 559 targets contienen `_SPLIT_` (xlsum); textos sin tildes (*esta*, *Paris*) | |
| E6 | Ambigüedad derivada de E2 | | *Junta los ingredientes* → sustantivo *junta* en lugar del verbo *juntar* |

### 3.2 Límites del método

| # | Límite | Evidencia |
|---|---|---|
| L1 | Cobertura de ARASAAC | Los OOV frecuentes no existen como keyword: *parte, forma, tipo, área, superficie, producto, material, probar, aplicar, tratar, necesario, posible, mezcla*. |
| L2 | Palabras de función | El 38 % de la salida son artículos y preposiciones (nivel 3). |
| L3 | Targets de varias oraciones | Secuencias de hasta 174 pictos sin marca de límite de oración. |
| L4 | Sin referencia humana | La calidad solo se ha evaluado visualmente; aún no hay PER/TER contra un conjunto de referencia. |

### 3.3 Lo que funciona bien

- Expresiones multipalabra: *lo siento*, *tiempo libre*, *objeto perdido*, *antes de*.
- Números.
- Negación en declarativas: *A Tom no le gusta hablar en francés* → `a <PER> le gustar <NEG> hablar en francés`.
- Verbos regulares en indicativo y tiempo pasado.

## 4. Hoja de ruta de mejoras

| Prioridad | Mejora | Ataca | Esfuerzo |
|---|---|---|---|
| A1 | Módulo de imperativos: tabla de irregulares, separación de enclíticos, verbo inicial en 2.ª persona → `<IMP>` | E1, E2, E6 | bajo |
| A2 | Filtro de NER: descartar entidades que empiezan con verbo o con una palabra que existe en el léxico como nombre común o verbo; buscar entidades de varias palabras en el léxico | E1 | bajo |
| A3 | Eliminar *re-*; *des-/in-/im-* solo con la misma categoría gramatical y lista de exclusiones | E3, E3b | bajo |
| A4 | Búsqueda sin acentos solo para términos de varias palabras | E4 | bajo |
| A5 | Limpiar `_SPLIT_`; limitar a 40 pictos o marcar límites de oración | E5, L3 | bajo |
| B1 | Respaldo con WordNet: lema OOV → synsets (OMW-es) → picto con el mismo synset o su hiperónimo (12,567 pictos tienen `synsets`) | L1 | medio |
| B2 | Respaldo morfológico: sustantivo→verbo (*mezcla*→*mezclar*), *-mente*→adjetivo, participio→verbo | L1 | bajo |
| B3 | Curación manual de los lemas OOV más frecuentes → `--extra_lexicon` | L1 | medio |
| C1 | WSD contextual: embeddings de la oración comparados con el campo `meaning` de cada candidato | E6 | medio |
| D1 | Alinear el formalismo (nivel, artículos, marcadores) con las convenciones del benchmark ImageCLEFtoPicto | L2 | decisión |
| E1' | Conjunto de referencia de 150–200 oraciones post-editadas para medir PER/TER de la gramática antes y después de cada mejora | L4 | medio |
| E2' | Post-edición con LLM restringida a los pictos candidatos del léxico, validada contra la referencia | todos | medio |
