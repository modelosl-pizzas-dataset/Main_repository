# Guía de comandos – Fase 2

Todos los comandos se ejecutan **desde la carpeta `Fase_2`**. Cada comando va en **una sola línea** (en el texto se muestran partidos solo cuando se indica). Ejemplos para PowerShell (Windows) y Bash; los comandos de Python son idénticos en ambos.

## 0. Preparar el entorno (una sola vez)

| Comando | Para qué sirve |
|---|---|
| `python -m venv .venv` | Crea un entorno virtual aislado (Python 3.12 recomendado). |
| `.venv\Scripts\Activate.ps1` (PowerShell) / `source .venv/bin/activate` (Bash) | Activa el entorno. Debe aparecer `(.venv)` en la terminal. |
| `pip install -r requirements.txt` | Instala las dependencias con las versiones exactas del proyecto (incluye pytest). |

> Si PowerShell bloquea la activación: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` y volver a activar.

## 1. Consultar la ayuda (`--help`)

```
python train.py --help
python predict.py --help
python prepare_data.py --help
```
Muestran todos los argumentos disponibles, su significado y valores por defecto. No ejecutan nada más.

## 2. Preparar los datos – `prepare_data.py`

```
python prepare_data.py --raw data/pizza_sales.csv --output data/dataset_optimizado.csv
```
Lee el CSV crudo y genera `data/dataset_optimizado.csv` (versión de Fase 2 de `analisis_sin_tiempo.py`): elimina identificadores, fechas, `quantity` y `total_price` (fuga de información) y deja `pizza_name` (solo como grupo de separación), `pizza_size`, `pizza_category`, `pizza_ingredients` y `unit_price`. No está en el repositorio: hay que ejecutarlo antes de entrenar (también antes de usar Docker, porque los datos se montan en solo lectura).

| Argumento | Descripción |
|---|---|
| `--raw` | CSV crudo (default `data/pizza_sales.csv`). |
| `--output` | CSV de salida (default `data/dataset_optimizado.csv`). |

## 3. Entrenar – `train.py`

```
python train.py --data data/dataset_optimizado.csv --target unit_price --output-dir models --version v1 --seed 42
```
Hace el proceso completo: lee y valida los datos → separa train/test **por tipo de pizza** (sin fuga) → ajusta el preprocesamiento solo con train → entrena Random Forest y el modelo base (regresión lineal) → evalúa ambos con la misma partición → guarda `models/<version>/modelo.joblib`, `metrics.json` y `metadata.json` e imprime un resumen.

| Argumento | Obligatorio | Descripción |
|---|---|---|
| `--data` | Sí | CSV de datos etiquetados. |
| `--version` | Sí | Identificador único de la versión (`v1`, `v2`, `v-docker`…). **Si ya existe, el script se detiene y no sobrescribe.** |
| `--target` | No | Variable objetivo (default `unit_price`). |
| `--output-dir` | No | Carpeta donde se crean las versiones (default `models`). |
| `--seed` | No | Semilla aleatoria (default 42). |
| `--test-size` | No | Fracción de prueba (default 0.2). |
| `--group-column` | No | Columna de grupo para separar sin fuga (default `pizza_name`). `none` = separación aleatoria (**no recomendado**: reproduce el error de la Fase 1). |
| `--n-estimators` | No | Número de árboles (default 200). |
| `--max-depth` | No | Profundidad máxima de los árboles (default sin límite). |
| `--git-commit` | No | Hash del commit a registrar en los metadatos (útil en Docker). |

**Crear las dos versiones de la entrega** (misma partición, distinta configuración):
```
python train.py --data data/dataset_optimizado.csv --target unit_price --output-dir models --version v1 --seed 42
python train.py --data data/dataset_optimizado.csv --target unit_price --output-dir models --version v2 --seed 42 --n-estimators 300 --max-depth 12
```

## 4. Predecir – `predict.py`

```
python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
```
Carga una versión guardada (no reentrena) y genera un CSV con las columnas de entrada **más** `prediccion` (precio unitario estimado), conservando el orden de las filas.

| Argumento | Obligatorio | Descripción |
|---|---|---|
| `--model` | Sí | Ruta a `modelo.joblib` **o** a la carpeta de la versión (`models/v2`). Así se elige la versión. |
| `--input` | Sí | CSV con `pizza_size`, `pizza_category`, `pizza_ingredients` (sin `unit_price`). |
| `--output` | Sí | CSV de salida (se crea la carpeta si no existe). |

Comportamiento: columnas en cualquier orden · columnas extra ignoradas (con aviso) · celdas nulas imputadas · categorías/ingredientes no vistos ignorados · tamaño fuera de `S, M, L, XL, XXL`, columna ausente o tipo numérico → **error claro, código de salida 1 y no se crea el archivo de salida**.

Predecir con la otra versión:
```
python predict.py --model models/v2 --input data/ejemplo_prediccion.csv --output outputs/predicciones_v2.csv
```

Comprobar el código de salida tras un error: `echo $?` (Bash) · `echo $LASTEXITCODE` (PowerShell).

## 5. Pruebas – `pytest`

```
python -m pytest
python -m pytest -v
```
Ejecuta las pruebas de `tests/test_aplicacion.py` (datos sintéticos, directorios temporales, sin red; tardan pocos segundos). `-v` lista cada prueba.

## 6. Docker

Los pasos completos (PowerShell y Bash) están en `DOCKER_PASOS.txt`. Resumen:
```
docker build -t modelos-fase2:1.0 .
```
y después `docker run --rm -v ...` con `python train.py ...` y `python predict.py ...` (ver archivo).
