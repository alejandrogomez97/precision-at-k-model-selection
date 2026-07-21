# Medium article — assets (in order of appearance)

Artículo: **`article_medium.md`** (narrativo, inglés, estilo Medium). **Todas las figuras y
tablas están EN INGLÉS.** Las tablas se han rasterizado a PNG con los números coloreados
(verde = mejor / rojo = peor por fila), porque Medium no admite tablas HTML con estilo al
pegar: se suben como imágenes igual que las figuras.

## Figuras (carpeta `assets/`, en orden de aparición)

| # | fichero | dónde va | qué muestra |
|---|---|---|---|
| 1 | `01_fig_A_story.png` | Setup / Chapter 1 | E1 vs E2 por fracción: equivalencia, diferencia pareada (el cruce) y efecto del desbalanceo |
| 2 | `02_fig_retrain.png` | Chapter 2 | Las 4 variantes (E1/E2 × retrain/no-retrain) y el efecto del reentrenamiento |
| 3 | `03_fig_hpo.png` | Chapter 3 | AP vs nº de configuraciones por método (grid/random/TPE/CMA-ES) y tiempo |
| 4 | `04_fig_C_isotime.png` | Chapter 4 | Ensemble estilo E1 vs E1 y estilo E2 vs E2 (AP, win-rate, tiempos) |
| 5 | `05_fig_isofinal.png` | Chapter 4 (equal-time) | A igualdad de tiempo: ens-E2 vs las versiones @t* |
| 6 | `06_fig_feateng.png` | Chapter 5 | AP vs % de variables inventadas al azar |

## Tablas (imágenes PNG coloreadas, verde = mejor / rojo = peor por fila)

| # | fichero | dónde va | qué muestra |
|---|---|---|---|
| 1 | `table1_e1_vs_e2.png` | Chapter 1 | E1 vs E2 por fracción (AP y tiempo) |
| 2 | `table2_hpo.png` | Chapter 3 | HPO: AP por presupuesto y método |
| 3 | `table3_isotime.png` | Chapter 4 | Igualdad de tiempo: E1/E1@t*/ens-E1/…/ens-E2 |
| 4 | `table4_feateng.png` | Chapter 5 | Feature engineering: AP vs % basura |

Se generan con `make_tables.py`. (`assets/tables.md` conserva las mismas tablas en markdown
plano, sin color, por si hicieran falta en texto.)

## Notas
- Los p-valores del artículo son t-test pareados sobre las mismas particiones.
- Figuras en inglés generadas con `FIGLANG=en` sobre los scripts `analyze_*.py` / `make_fig_A.py`.
