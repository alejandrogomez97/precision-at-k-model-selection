# ¿Un solo conjunto de validación, o dos? Un estudio empírico sobre datos desbalanceados

> Borrador generado durante la ejecución del experimento. Las secciones de
> resultados marcadas con «⟨…⟩» se rellenan automáticamente al terminar cada fase.

## 1. La pregunta

En un artículo previo defendí, sobre todo con argumentos teóricos, que **un único
conjunto de validación basta** para hacer a la vez *early stopping* (elegir el
número de árboles/iteraciones) y selección de hiperparámetros: el early stopping
no es más que preseleccionar sobre un hiperparámetro más, así que separar dos
conjuntos responde a limitaciones de datos, no a una necesidad metodológica.

Aquí lo pongo a prueba **empíricamente** sobre decenas de datasets binarios
desbalanceados, midiendo qué pasa **en función de la cantidad de datos**. La
intuición a falsar es doble y aparentemente contradictoria:

- Con **muchos datos**, reservar un conjunto de validación independiente para la
  *selección* debería reducir la **maldición del ganador** (el sesgo optimista de
  elegir el mejor entre muchos candidatos evaluados sobre los mismos datos) y por
  tanto generalizar mejor.
- Con **pocos datos**, ese conjunto reservado es diminuto y ruidoso; usar toda la
  información vía validación cruzada (CV) para decidirlo todo debería salir a
  cuenta.

## 2. Las dos estrategias

Ambas parten del mismo *pool* de desarrollo `dev` y del **mismo conjunto de test**
(fijo y común en cada punto), y comparten el mismo espacio de candidatos. La
**única** diferencia es el mecanismo de selección, de modo que el experimento
aísla el efecto de la maldición del ganador frente al ruido de selección.

**E1 — validación separada.** `dev → train + val`. El número de árboles de cada
candidato se fija por **CV dentro de `train`** (early stopping). El **mejor
candidato** se elige con el conjunto **`val` independiente** (que no se usó para el
early stopping). El modelo final se **reentrena en `train`+`val` (= `dev`)** y se
evalúa en test.

**E2 — solo CV.** `dev → CV`. Los folds *out-of-fold* (OOF) deciden **a la vez** el
número de árboles (early stopping) **y** el mejor candidato. El modelo final se
reentrena en `dev` y se evalúa en test.

Como en E1 el modelo final también se reentrena con todo `dev`, ambas estrategias
usan **la misma cantidad de datos** para el modelo final: lo que cambia es de dónde
sale la señal de selección (un `val` insesgado pero pequeño, frente a una
estimación OOF que usa todos los datos pero reutiliza lo mismo para elegir árboles
y modelo).

## 3. Montaje experimental

- **Datos:** datasets binarios desbalanceados (imblearn + OpenML) ya materializados
  del estudio *precision-at-k*. Prevalencias e índices de desbalanceo variados.
- **Test fijo:** 30 % del dataset (capado a 6.000 filas), idéntico para E1 y E2 en
  cada punto. Solo encoge el *pool* de desarrollo.
- **Barrido de tamaño:** `dev ∈ {300, 600, 1200, 2500, 5000, 10000, 20000}`
  (submuestreo estratificado), para ver la performance en función de los datos.
- **Repeticiones:** varias semillas por punto; se promedia.
- **Modelo:** LightGBM. Grid de 6 combinaciones (`num_leaves`×`reg_lambda`) × 3
  técnicas de desbalanceo (`none`, `class_weight`, `SMOTE`) = 18 candidatos.
- **Early stopping:** métrica `average_precision`, 50 rondas de paciencia,
  hasta 3000 árboles.
- **Métrica primaria:** Average Precision (PR-AUC), la adecuada en desbalanceo;
  se registra también logloss. Preprocesado ajustado por-fold (sin fugas).
- Se registran **tiempos** de todo.

## 4. Resultado principal: ¿cuándo gana cada estrategia?

Con **93 datasets, 3 semillas y 1.127 comparaciones pareadas**, el resultado es
tan claro como sobrio: **las dos estrategias son prácticamente equivalentes**.

- La diferencia mediana de AP entre E1 y E2 es de **0.0004**; en el **54 %** de las
  celdas es menor de 0.01 de AP. A efectos prácticos, da igual cuál uses.
- Existe, eso sí, una **ventaja pequeña pero sistemática y estadísticamente
  detectable para E2** (un solo *split*, la CV decide árboles y modelo):
  ΔAP medio = **−0.0024** a favor de E2, *t* = −2.42, **p = 0.016**; win-rate de
  E1 = 0.446.
- Esa ventaja de E2 **crece con el desbalanceo**: de ~0 con índice < 5 a
  **−0.0037** con índice > 20.
- Y es mayor con **pocos datos**: en `dev = 300` la diferencia es −0.0056
  (≈ 2.3 errores estándar), la más grande del barrido.
- **El régimen que anticipábamos —E1 (val separado) ganando con muchos datos por
  menor maldición del ganador— no aparece.** E1 no supera a E2 de forma clara en
  ningún tamaño.

| tamaño dev | AP E1 | AP E2 | ΔAP (E1−E2) | win-rate E1 | n |
|---:|---:|---:|---:|---:|---:|
| 300 | 0.502 | 0.507 | −0.0056 | 0.39 | 265 |
| 600 | 0.540 | 0.541 | −0.0002 | 0.44 | 268 |
| 1200 | 0.596 | 0.597 | −0.0008 | 0.54 | 226 |
| 2500 | 0.593 | 0.597 | −0.0038 | 0.43 | 163 |
| 5000 | 0.553 | 0.555 | −0.0018 | 0.46 | 96 |
| 10000 | 0.536 | 0.539 | −0.0025 | 0.39 | 62 |
| 20000 | 0.453 | 0.454 | −0.0006 | 0.47 | 47 |

*(El AP absoluto baja a partir de 5.000 no porque el modelo empeore, sino porque
solo los datasets grandes —y más difíciles— alcanzan esos tamaños: es un efecto de
composición de la muestra. La comparación válida es la **pareada**, dentro de cada
dataset.)*

**Lectura.** Reservar un conjunto de validación independiente para la selección
(E1) no compra nada aquí: en este diseño el modelo final se reentrena con todo
`dev` en ambas estrategias, así que E1 solo aporta una señal de selección
*insesgada pero más pequeña y ruidosa*, mientras que E2 usa toda la información vía
OOF. La supuesta penalización por maldición del ganador de E2 —reutilizar los
mismos folds para elegir árboles y modelo— resulta **despreciable** en la práctica
con 18 candidatos, y no compensa el coste de trocear los datos. Cuanto más escaso o
desbalanceado es el problema (menos positivos en el `val` de E1), más se nota.

*Figuras:* `fig_A_story.png` (3 paneles: equivalencia, ventaja vs tamaño, ventaja
vs desbalanceo), `fig_ap_vs_size.png` y `fig_diff_vs_size.png`.

## 5. Fase B — ¿mejora Optuna al grid?

Se repite el montaje sustituyendo el grid por **Optuna (TPE, 40 trials)** sobre un
espacio de hiperparámetros más amplio de LightGBM, en ambas estrategias. Se compara
AP en test **y** tiempo de búsqueda.

⟨RESULTADOS_B⟩

*Figura:* `fig_B_optuna.png`.

## 6. Fase C — muchas familias + ensembles: ¿mejor y más rápido?

En lugar de muchos LightGBM, se entrena un **pool de familias** (regresión
logística, RF, ExtraTrees, HistGBM, LightGBM, XGBoost, CatBoost, MLP, kNN, GNB) y
se construye un **ensemble por selección greedy de Caruana** sobre `val`. Se mide la
curva *anytime*: mejor AP en test alcanzado en función del **tiempo acumulado**, y
se compara con la referencia E1-grid (su AP y su tiempo de búsqueda). Pregunta:
¿se **supera** a E1 y **se consolida en test**, y en **menos tiempo**?

⟨RESULTADOS_C⟩

*Figura:* `fig_C_families.png`.

## 7. Conclusiones

⟨CONCLUSIONES⟩
