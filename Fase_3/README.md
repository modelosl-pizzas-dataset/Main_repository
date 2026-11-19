# Fase 3 – API REST del modelo de precio unitario de pizzas

**Proyecto Integrador – Modelos y Simulación de Sistemas I (2026 II)**
**Integrantes:** _(completar)_ · **Entrega – commit/etiqueta de Git:** _(completar, p. ej. `entrega-3`)_

La API (Flask) permite consultar el modelo de la Fase 2 y entrenar nuevas versiones mediante solicitudes HTTP. **No reimplementa nada**: `/predict` y `/train` llaman a las mismas funciones que `predict.py` y `train.py` (`src/`), con el preprocesamiento guardado dentro de cada versión.

## 1. Contenido de la carpeta

```
Fase_3/
├── api.py                  # Aplicación Flask: rutas y manejo de errores
├── src/                    # Código de la Fase 2 + src/service.py (validación JSON, modelo activo, versiones) y src/settings.py (configuración)
├── train.py  predict.py    # CLI de la Fase 2 (siguen funcionando)
├── client.py               # Cliente de prueba (Python, solo librería estándar)
├── comandos_curl.md        # Mismas solicitudes con curl / PowerShell
├── tests/                  # test_api.py (API) + test_aplicacion.py (Fase 2) + conftest.py
├── data/                   # dataset_optimizado.csv, ejemplo_prediccion.csv, ejemplo_solicitud.json
├── models/                 # v1/ y v2/ (modelo.joblib, metrics.json, metadata.json)
├── requirements.txt  Dockerfile  .dockerignore  .gitignore  DOCKER_PASOS.txt
```

## 2. Puesta en marcha local

Python 3.12 o superior (probado con 3.12 y 3.13). Desde `Fase_3`, en la terminal de VS Code (PowerShell):

```
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m flask --app api run --host 0.0.0.0 --port 8000
```

La API queda en `http://localhost:8000`. Con el servidor activo, en **otra terminal** (con el entorno activado) ejecute el cliente: `python client.py`.

Servidor de desarrollo de Flask: suficiente para la demostración local; para producción debería reemplazarse por un servidor WSGI.

## 3. Configuración y selección de la versión del modelo

La API predice con **una versión activa**, elegida con variables de entorno (valores por defecto entre paréntesis):

| Variable | Uso |
|---|---|
| `MODEL_VERSION` (`v1`) | Versión que atiende `/predict`. |
| `MODELS_DIR` (`models`) | Carpeta con las versiones `models/<version>/`. |
| `DATA_PATH` (`data/dataset_optimizado.csv`) | Datos con los que entrena `/train`. |
| `TARGET` (`unit_price`), `GROUP_COLUMN` (`pizza_name`), `TRAIN_SEED` (`42`), `TEST_SIZE` (`0.2`) | Configuración de la Fase 2 usada por `/train`. |
| `CORS_ORIGINS` (`*`) | Origen permitido para un futuro frontend (vacío = desactivado). |

Las rutas relativas se resuelven respecto a la carpeta de `api.py`. **Cambiar de versión** = reiniciar con otra `MODEL_VERSION`:

```
$env:MODEL_VERSION="v2"; python -m flask --app api run --port 8000      # PowerShell
MODEL_VERSION=v2 python -m flask --app api run --port 8000              # Bash
```

Versiones disponibles en un momento dado: `GET /models` o las carpetas de `models/`. Si la versión configurada no existe, `/predict` responde **503** indicando cuáles hay disponibles y cómo elegir una (ver sección 5).

## 4. Endpoints

| Método y ruta | Descripción | Respuestas |
|---|---|---|
| `GET /health` | Indica que la API está en ejecución y si el modelo activo está disponible. | 200 |
| `POST /predict` | Predice `unit_price` para un registro JSON. | 200 · 400 · 422 · 503 |
| `POST /train` | Entrena y guarda una nueva versión (cuerpo vacío o con parámetros opcionales). | 201 · 400 · 409 · 422 · 500 |
| `GET /models` | Versiones guardadas, métricas y cuál es la activa. | 200 |
| `GET /schema` | Variables de entrada, tipos y valores permitidos/conocidos (útil para un formulario futuro). | 200 · 503 |
| `GET /` | Resumen del servicio y de sus rutas. | 200 |

Los errores siempre devuelven JSON: `{"error": "<mensaje>", "tipo": "...", "campos": ["..."]}`.

### 4.1 `POST /predict`

Cada columna predictora del CSV de la Fase 2 es una clave JSON; todas son **texto** (o `null` para un dato faltante, que se imputa con la regla guardada). La variable objetivo **no** es necesaria. Las claves adicionales se ignoran y se informan en `campos_ignorados`.

| Campo | Tipo | Regla |
|---|---|---|
| `pizza_size` | texto o `null` | `S`, `M`, `L`, `XL`, `XXL`; otro valor → 422. |
| `pizza_category` | texto o `null` | `Classic`, `Veggie`, `Supreme`, `Chicken`; una categoría nueva se ignora. |
| `pizza_ingredients` | texto o `null` | Ingredientes separados por coma; los no vistos se ignoran. |

Solicitud (fila de `data/ejemplo_prediccion.csv`) y respuesta reales:

```
POST /predict
{"pizza_size": "M", "pizza_category": "Classic", "pizza_ingredients": "Pepperoni, Mushrooms, Red Onions, Red Peppers, Bacon"}

HTTP 200
{"prediccion": 16.0, "version": "v1"}
```

Con un campo ausente (se identifica en el mensaje):

```
POST /predict
{"pizza_size": "S", "pizza_category": "Classic"}

HTTP 422
{"error": "Falta el campo obligatorio 'pizza_ingredients'.", "tipo": "validacion", "campos": ["pizza_ingredients"]}
```

### 4.2 `POST /train`

Reutiliza `src.training.train` con la configuración de la Fase 2 (`DATA_PATH`, `TARGET`, semilla, partición por grupos). **Cuerpo vacío válido**: genera el siguiente identificador libre (`v3` si existen `v1` y `v2`) y responde 201 al terminar (tarda ~10 s con los datos completos). Parámetros opcionales: `version`, `seed`, `n_estimators`, `max_depth`, `test_size`.

```
POST /train

HTTP 201
{"version": "v3", "mensaje": "Versión entrenada y guardada. Las versiones anteriores se conservaron.",
 "metricas": {"modelo": {"rmse": 1.1855, ...}, "modelo_base": {...}, "referencia_media": {...}},
 "archivos": {"modelo": "models/v3/modelo.joblib", "metricas": "models/v3/metrics.json", "metadatos": "models/v3/metadata.json"},
 "version_activa": "v1",
 "nota": "La API sigue prediciendo con la versión activa. Para usar la nueva, reinicie con MODEL_VERSION=v3."}
```

La nueva versión **no** se activa sola: la API sigue usando la versión configurada hasta que se reinicie con otra `MODEL_VERSION`. Un identificador existente devuelve 409 sin sobrescribir; un entrenamiento simultáneo también devuelve 409.

## 5. Códigos de error

| Código | Cuándo |
|---|---|
| 400 | El cuerpo no es JSON válido. |
| 422 | Falta un campo obligatorio, tipo distinto de texto/`null`, tamaño no permitido o parámetros de `/train` inválidos. El mensaje nombra el campo. |
| 503 | La versión configurada no está disponible. Ejemplo: `El modelo configurado 'v99' no está disponible en 'models'. Versiones disponibles: v1, v2. Seleccione una con la variable de entorno MODEL_VERSION y reinicie la API.` |
| 409 | `/train` con identificador repetido o con un entrenamiento en curso. |
| 500 | Error interno o datos de entrenamiento no encontrados (`DATA_PATH`). |

## 6. Cliente de prueba

```
python client.py                       # /health, /models y una predicción por cada fila de data/ejemplo_prediccion.csv
python client.py --incluir-invalida    # además, un registro sin 'pizza_ingredients' (debe dar 422)
python client.py --train               # además, POST /train
```

También hay solicitudes equivalentes con `curl` y PowerShell en `comandos_curl.md`.

## 7. Docker

Pasos completos (PowerShell y Bash) en **`DOCKER_PASOS.txt`**. Resumen desde `Fase_3`:

```
docker build -t modelos-fase3:1.0 .
docker run --rm -p 8000:8000 -v "$PWD/data:/app/data:ro" -v "$PWD/models:/app/models" modelos-fase3:1.0
```

- **Puerto:** 8000 (`-p 8000:8000`); el servidor escucha en `0.0.0.0`.
- **Datos:** `data/` montada de solo lectura en `/app/data` (la usa `/train`).
- **Modelos:** `models/` montada en `/app/models` (lectura y escritura): las versiones de `/train` persisten en su equipo.
- **Versión activa:** `-e MODEL_VERSION=v2`.

## 8. Pruebas

```
python -m pytest
```

`tests/test_api.py` usa el cliente de pruebas de Flask y un **modelo pequeño** entrenado con datos sintéticos en carpetas temporales (sin red). Las tres pruebas exigidas son `test_health_responde_200_y_estado`, `test_predict_entrada_valida_devuelve_prediccion_y_version` y `test_predict_con_campo_obligatorio_ausente_se_rechaza`. Además cubren tipos incompatibles, tamaño no permitido, nulos, campos extra, JSON inválido, modelo no disponible (503), cambio de versión, `/train` (versión nueva, conservar anteriores, repetido, parámetros inválidos, datos ausentes), `/models`, `/schema` y CORS. `tests/test_aplicacion.py` conserva las 12 pruebas de la Fase 2.

## 9. Preparada para un frontend

Los endpoints ya son consumibles desde un navegador: CORS habilitado, errores en JSON con `campos` para marcar el campo erróneo, `GET /schema` para construir el formulario (tamaños permitidos, categorías e ingredientes conocidos) y `GET /models` para mostrar la versión activa y las métricas. No se implementó ningún frontend.

## 10. Evidencia de ejecución (validación del paquete)

Salidas reales obtenidas al ejecutar `client.py` contra la API iniciada (`MODEL_VERSION=v1`). Reemplácelas con las de su propia ejecución:

```
GET /health          -> HTTP 200 {"status": "ok", "servicio": "api en ejecución", "modelo_activo": "v1", "modelo_disponible": true}
POST /predict (S, Classic, "Sliced Ham, Pineapple, Mozzarella Cheese")   -> HTTP 200 {"prediccion": 10.5, "version": "v1"}
POST /predict (M, Classic, "Pepperoni, Mushrooms, Red Onions, ...")      -> HTTP 200 {"prediccion": 16.0, "version": "v1"}
POST /predict (sin pizza_ingredients)                                    -> HTTP 422 {"error": "Falta el campo obligatorio 'pizza_ingredients'.", ...}
POST /train          -> HTTP 201 {"version": "v3", ...}   # ~10 s; models/v3/ con modelo.joblib, metrics.json y metadata.json; v1 y v2 intactas
```

Secuencia de comprobación: (01) `docker build` y `docker run` de la sección 7; (02) `python client.py` y revisar predicción y versión; (03) `python client.py --train`, comprobar `models/v3/` en su equipo y ejecutar `python -m pytest`.

_Pegar aquí la salida de `docker run`, de `client.py` y de `python -m pytest` en su equipo._
