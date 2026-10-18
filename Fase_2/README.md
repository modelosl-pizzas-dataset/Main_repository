# Fase 2 – Predicción del precio unitario de pizzas (Scripts y Docker)

**Proyecto Integrador – Modelos y Simulación de Sistemas I (2026 II)**
**Integrantes:** _(completar nombres)_
**Entrega 2 – commit/etiqueta de Git:** _(completar: p. ej. etiqueta `entrega-2` y hash del commit)_

## 1. Problema y relación con la Fase 1

Se mantiene el problema aprobado: predecir `unit_price` (precio unitario de una pizza, regresión) a partir de su tamaño, categoría e ingredientes, con el dataset *Pizza Sales* (48.620 pedidos, 2015). El algoritmo elegido en la Fase 1 fue **Random Forest**.

La Fase 2 convierte ese trabajo en una aplicación reutilizable: `prepare_data.py` (datos), `train.py` (entrenamiento + versionamiento) y `predict.py` (predicción), con la lógica compartida en `src/`, más pruebas con pytest y una imagen Docker. La carpeta `Fase_1` se conserva (limpia, ver `LIMPIEZA_FASE_1.md`); en ella solo quedó el código que genera el dataset y el modelo elegido.

## 2. Correcciones respecto a la Fase 1

| # | Problema en la Fase 1 | Corrección en la Fase 2 |
|---|---|---|
| 1 | **Fuga de información por duplicados.** El dataset tiene 48.620 filas pero solo 91 combinaciones distintas (pizza + tamaño) y cada una tiene un único precio. Con `train_test_split` aleatorio, las mismas pizzas caen en train y test, por eso la Fase 1 reportaba RMSE = 0.0 y R² = 1.0 (memorización, no capacidad de generalizar). | La separación es por **grupos naturales**: `GroupShuffleSplit` por `pizza_name` (32 tipos). Un tipo de pizza completo está en entrenamiento **o** en prueba, nunca en ambos. La métrica ahora mide predicción sobre pizzas no vistas. |
| 2 | `LabelEncoder`, vocabulario de ingredientes (one-hot) e imputación por media se calculaban con **todos** los datos antes de separar. | Todo el preprocesamiento vive en un `Pipeline`/`ColumnTransformer` que se ajusta **solo con train** y se guarda junto al modelo. |
| 3 | `pizza_size` codificado con `LabelEncoder` (orden alfabético: L, M, S, XL, XXL), sin significado ordinal. | Codificación ordinal explícita `S < M < L < XL < XXL`. |
| 4 | `quantity` se usaba como variable, pero no influye en el precio unitario. | Se elimina. |
| 5 | `dataset_optimizado.csv` era una copia del dataset sin tiempo (≈7 MB, 70 columnas). | `prepare_data.py` genera un CSV limpio de 5 columnas con transformaciones **sin estado**; el estado (vocabulario, codificadores) lo aprende el Pipeline. |
| 6 | Sin modelo base en la comparación final, rutas relativas frágiles, sin validación ni manejo de errores. | Se evalúa un modelo base (regresión lineal) con la misma partición y métrica, argumentos por línea de comandos, validación de entradas y códigos de salida. |

## 3. Entorno local

- **Python 3.12** (el mismo que usa la imagen Docker). Dependencias con versiones exactas en `requirements.txt`.
- Los comandos son para ejecutarse desde la carpeta `Fase_2`. Terminal usada por el equipo: _(completar: PowerShell / Git Bash)_.

```
python -m venv .venv
.venv\Scripts\Activate.ps1        # PowerShell      |   source .venv/bin/activate   # Bash
pip install -r requirements.txt
```

## 4. Datos

- **Fuente:** dataset *Pizza Sales* usado en la Fase 1 (CSV de pedidos de pizza de 2015). Se incluye en este repositorio como `Fase_2/data/pizza_sales.csv`. Si hubiera que obtenerlo de nuevo, debe descargarse de la fuente original usada en la Fase 1 (_completar enlace_; 12 columnas, 48.620 filas) y guardarse con ese nombre en `Fase_2/data/`. No se requieren credenciales para ejecutar la aplicación.
- **Archivos en `data/`:**
  - `pizza_sales.csv`: CSV crudo (entrada de `prepare_data.py`).
  - `dataset_optimizado.csv`: dataset etiquetado para entrenar (generado con `python prepare_data.py`; ya incluido).
  - `ejemplo_prediccion.csv`: ejemplo pequeño (7 filas, sin variable objetivo) para `predict.py`.

**Esquema del CSV de predicción** (en cualquier orden; columnas extra se ignoran):

| Columna | Tipo | Valores / reglas |
|---|---|---|
| `pizza_size` | texto | `S`, `M`, `L`, `XL`, `XXL`. Cualquier otro valor → error. Nulo → se imputa con la moda de train. |
| `pizza_category` | texto | `Classic`, `Veggie`, `Supreme`, `Chicken`. Categoría nueva → se ignora (todas las columnas one-hot en 0). Nulo → moda de train. |
| `pizza_ingredients` | texto | Ingredientes separados por coma, p. ej. `"Sliced Ham, Pineapple, Mozzarella Cheese"`. Ingredientes no vistos → se ignoran. Nulo → sin ingredientes y `n_ingredientes` = mediana de train. |

**Salida de `predict.py`:** el mismo CSV de entrada (mismo orden de filas) + la columna `prediccion` = precio unitario estimado (misma unidad que `unit_price`).

## 5. Comandos principales

Detalle completo de cada comando y argumento en **`COMANDOS.md`**.

```
python prepare_data.py --raw data/pizza_sales.csv --output data/dataset_optimizado.csv
python train.py --data data/dataset_optimizado.csv --target unit_price --output-dir models --version v1 --seed 42
python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
python train.py --help
python predict.py --help
```

## 6. Estructura y ubicación de resultados

```
Fase_2/
├── train.py  predict.py  prepare_data.py     # puntos de entrada (CLI con argparse)
├── src/                                      # lógica reutilizable (config, data, features, training, prediction, versioning, errors)
├── tests/                                    # conftest.py + test_aplicacion.py
├── data/                                     # pizza_sales.csv, dataset_optimizado.csv, ejemplo_prediccion.csv
├── models/v1/  models/v2/                    # modelo.joblib + metrics.json + metadata.json por versión
├── outputs/                                  # predicciones generadas (no se versionan en Git)
├── requirements.txt  Dockerfile  .dockerignore  .gitignore
└── README.md  COMANDOS.md  DOCKER_PASOS.txt
```

**Versionamiento del modelo.** Cada ejecución de `train.py` crea `models/<version>/`. Si el identificador ya existe, el script termina con error y **no sobrescribe**; la carpeta se escribe primero en un directorio temporal y se renombra al final, para no dejar versiones a medias. `predict.py --model models/v2` elige la versión.

| Versión | Configuración | Qué contiene |
|---|---|---|
| `v1` | Random Forest, 200 árboles, sin límite de profundidad, semilla 42 | `modelo.joblib`, `metrics.json`, `metadata.json` |
| `v2` | Random Forest, 300 árboles, `max_depth=12`, semilla 42 | ídem |

`modelo.joblib` guarda un único objeto con el `Pipeline` completo (preprocesamiento ajustado + modelo) y el esquema de entrada, por lo que `predict.py` no necesita nada más. `metadata.json` incluye versión, fecha, tipo de problema, variable objetivo, algoritmo, variables y tipos, parámetros, semilla, datos usados (archivo, SHA-256, nº de observaciones), estrategia y tamaños de separación, métrica, versión de Python y de librerías, archivo de dependencias y hash del commit (`no-disponible` si se entrena fuera de un repositorio Git o sin `--git-commit`; reentrenar tras el primer commit lo registra).

## 7. Docker

Pasos completos, con montajes y variantes para PowerShell y Bash, en **`DOCKER_PASOS.txt`**. Resumen (Bash, desde `Fase_2`):

```
docker build -t modelos-fase2:1.0 .
mkdir -p models outputs
docker run --rm -v "$PWD/data:/app/data:ro" -v "$PWD/models:/app/models" modelos-fase2:1.0 python train.py --data data/dataset_optimizado.csv --target unit_price --output-dir models --version v-docker --seed 42
docker run --rm -v "$PWD/data:/app/data:ro" -v "$PWD/models:/app/models:ro" -v "$PWD/outputs:/app/outputs" modelos-fase2:1.0 python predict.py --model models/v-docker/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones_docker.csv
```

La imagen usa `python:3.12.3-slim-bookworm`, instala dependencias, copia solo el código (`src/`, `tests/`, scripts) y **no entrena durante el build**. Datos, modelos y resultados se montan como volúmenes y persisten en el equipo anfitrión.

## 8. Pruebas

```
python -m pytest
```

`tests/test_aplicacion.py` (datos sintéticos, `tmp_path`, sin red):

| Prueba | Qué verifica |
|---|---|
| `test_entrenamiento_genera_artefacto_metricas_y_metadatos` | Se generan modelo cargable, `metrics.json` (modelo y base) y `metadata.json` con los campos mínimos. |
| `test_separacion_por_grupos_sin_fuga` | Ningún tipo de pizza queda en train y test a la vez. |
| `test_prediccion_conserva_resultados_del_modelo_guardado` | Una predicción por fila, mismo orden, y valores iguales a los del modelo antes de guardarlo. |
| `test_prediccion_no_depende_del_orden_de_columnas_ni_de_columnas_extra` | Columnas desordenadas/extra dan los mismos resultados. |
| `test_valores_faltantes_se_imputan_sin_eliminar_filas` | Nulos en las tres variables se imputan; no se pierden filas. |
| `test_categoria_e_ingrediente_no_vistos_se_ignoran` | Categorías/ingredientes nuevos no rompen la predicción. |
| `test_columna_requerida_ausente_da_error_claro` | `DataValidationError` con el nombre de la columna y sin archivo de salida. |
| `test_cli_predict_con_error_termina_con_codigo_distinto_de_cero` | `predict.py` sale con código ≠ 0 ante columna ausente o modelo inexistente. |
| `test_tipo_o_tamano_incompatible_da_error` | Tipo numérico o tamaño inválido producen error claro. |
| `test_segunda_version_conserva_la_primera_y_ambas_cargan` | `v1` y `v2` coexisten y cargan. |
| `test_identificador_repetido_no_sobrescribe` | Un identificador repetido falla y deja `v1` intacta. |
| `test_entrenamiento_sin_variable_objetivo_falla_con_mensaje_claro` | Falta de `unit_price` en el CSV de entrenamiento → error claro. |

## 9. Resultados, comparación con el modelo base y limitaciones

Métricas sobre prueba (10.344 filas, 7 tipos de pizza no vistos; 38.276 filas y 25 tipos en entrenamiento), semilla 42:

| Modelo | RMSE | MAE | R² |
|---|---|---|---|
| Random Forest `v1` | 1.1855 | 0.7699 | 0.8238 |
| Random Forest `v2` | 1.1773 | 0.7640 | 0.8262 |
| **Base:** regresión lineal | **1.1049** | 0.9361 | 0.8469 |
| Referencia: predecir la media | 3.0678 | 2.6335 | −0.1800 |

Interpretación y limitaciones conocidas:

- **El Random Forest no supera al modelo base en RMSE** con esta partición (sí en MAE). Los R² ≈ 0.82–0.85 reemplazan al 1.0 de la Fase 1, que era un artefacto de la fuga. Se conserva Random Forest por decisión del equipo; no se exigía mejorar la métrica.
- **Alta sensibilidad a la partición:** con solo 32 tipos de pizza, qué 7 caen en prueba cambia mucho el resultado (con `--seed 1` el RMSE del Random Forest es 1.51 frente a 1.22 de la regresión lineal; con `--seed 7`, 2.90 frente a 2.69). Las métricas deben leerse como una estimación ruidosa.
- El precio depende casi por completo del tamaño y de la carta del restaurante (en la Fase 1, `pizza_size` concentraba ≈79 % de la importancia); el modelo predice un precio "de carta" y no sirve para otro restaurante.
- `pizza_size` solo admite `S, M, L, XL, XXL`; `pizza_category` nueva se ignora, lo que puede restar precisión.
- **Reproducibilidad:** Random Forest usa la semilla (`random_state`) y los resultados coinciden entre ejecuciones con las mismas versiones de librerías; pueden aparecer diferencias numéricas mínimas entre sistemas operativos o versiones.
- El modelo guardado se ajusta con el conjunto de entrenamiento (80 %); no se reentrena con todos los datos para que las métricas reportadas correspondan exactamente al artefacto.

## 10. Evidencia de ejecución

Salida real de `train.py` (`v1`):

```
Versión v1 entrenada | train=38276 test=10344 | GroupShuffleSplit por 'pizza_name'
Modelo                       RMSE      MAE       R2
Random Forest              1.1855   0.7699   0.8238
Regresión lineal (base)    1.1049   0.9361   0.8469
Media (referencia)         3.0678   2.6335  -0.1800
```

`predict.py` sobre `data/ejemplo_prediccion.csv` (extracto de `outputs/predicciones.csv`; las dos últimas filas tienen nulos):

```
pizza_size,pizza_category,pizza_ingredients,prediccion
S,Classic,"Sliced Ham, Pineapple, Mozzarella Cheese",10.5
M,Classic,"Pepperoni, Mushrooms, Red Onions, Red Peppers, Bacon",16.0
L,Classic,,20.5
,Veggie,"Mushrooms, Tomatoes, Garlic",17.9225
```

Error previsible (columna ausente): `ERROR: Faltan columnas requeridas: ['pizza_ingredients']...` con código de salida 1 y sin archivo de salida.

_Evidencia de Docker y de pytest en el equipo del grupo: (pegar aquí la salida de `docker build`, de los dos `docker run` y de `python -m pytest`)._

## 11. Fuentes y código externo

- Documentación de scikit-learn (Pipeline, ColumnTransformer, GroupShuffleSplit, RandomForestRegressor) y de pandas.
- Dataset *Pizza Sales* usado en la Fase 1 (_completar fuente/enlace_).
- No se adaptó código externo; el código de `Fase_1` es del equipo.
