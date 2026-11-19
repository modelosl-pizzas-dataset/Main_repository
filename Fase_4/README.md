# Fase 4 – Monitoreo de la API

**Proyecto Integrador – Modelos y Simulación de Sistemas I (2026 II)**
**Integrantes:** _(completar)_ · **Entrega – commit/etiqueta de Git:** _(completar, p. ej. `entrega-4`)_

Esta carpeta conserva la API de la Fase 3 (ver `../Fase_3/README.md` para el detalle de `/health`, `/predict`, `/train`, `/models`, `/schema` y sus códigos de error) y agrega:

1. **Registro persistente** de cada solicitud a `POST /predict` (`src/request_log.py`, integrado en `api.py`).
2. **Script de monitoreo** `monitor.py` (lógica en `src/monitoring.py`) con total de solicitudes y porcentaje de fallos.
3. **Política de reentrenamiento** y versiones identificadas (más abajo).

## 1. Contenido nuevo respecto a la Fase 3

```
Fase_4/
├── api.py                     # + registro de solicitudes (after_request) y versión activa por defecto v2
├── monitor.py                 # Script de monitoreo
├── src/request_log.py         # Escribe una fila por solicitud (CSV en modo agregar)
├── src/monitoring.py          # Lectura del registro y cálculo de métricas
├── logs/                      # Aquí se crea predict_log.csv (registro persistente; vacío en la entrega)
├── docs/muestra_registro.csv  # Muestra del registro usado en el reporte
├── tests/test_monitoreo.py    # Pruebas del registro y del monitoreo (+ test_api.py y test_aplicacion.py heredadas)
└── (resto igual que Fase_3: client.py, data/, models/ con v1 y v2, requirements.txt, Dockerfile, ...)
```

## 2. Registro de solicitudes

- **Qué se registra:** una fila por cada `POST /predict`, **incluidas las rechazadas** por validación (422), JSON inválido (400), modelo no disponible (503) o errores internos (500). Se escribe al conocer el resultado, en el gancho `after_request` de Flask, que se ejecuta una sola vez por solicitud: cada intento genera una sola fila.
- **Campos:** `fecha_hora` (ISO 8601, hora local del servidor; en Docker, UTC), `estado_http`, `version` (versión que produjo la respuesta; vacía en errores) y `prediccion` (vacía en errores).
- **Persistencia:** el archivo se abre en modo agregar y el encabezado se escribe solo al crearlo, por lo que el historial se conserva entre ejecuciones. Un fallo al escribir el registro no interrumpe la API.
- **Ubicación:** `Fase_4/logs/predict_log.csv` (variable de entorno `LOG_PATH`). En Docker, la carpeta `logs/` se monta como volumen y persiste en el equipo.
- Solo se registra `POST /predict`; `/health`, `/models`, `/schema` y `/train` no generan filas.

Ejemplo del archivo (`docs/muestra_registro.csv`):

```
fecha_hora,estado_http,version,prediccion
2026-11-19T22:54:46,200,v2,10.5
2026-11-19T22:54:46,200,v2,16.0
2026-11-19T22:54:46,422,,
```

## 3. Comandos

Desde `Fase_4`, con el entorno virtual activado (`py -m venv .venv`, `.venv\Scripts\Activate.ps1`, `pip install -r requirements.txt`):

```
python -m flask --app api run --host 0.0.0.0 --port 8000     # Terminal 1: inicia la API con el registro habilitado (versión activa v2)
python client.py --filas 2 --incluir-invalida                # Terminal 2: 2 solicitudes válidas + 1 con un campo ausente
python monitor.py --log logs/predict_log.csv                 # Terminal 2: reporte de operación
python -m pytest                                             # Pruebas
```

Para repetir el ensayo desde un registro vacío, deténgase la API (Ctrl+C) y borre o renombre `logs/predict_log.csv` (`Remove-Item logs\predict_log.csv` en PowerShell). Con Docker (comando de inicio, montajes y ubicación del registro): ver `DOCKER_PASOS.txt`.

**Qué muestra `monitor.py`:** total de solicitudes (filas de datos del registro), solicitudes fallidas (HTTP 4xx o 5xx), porcentaje de fallos (con coma decimal) y el desglose por versión. Con el registro vacío muestra total 0 y «sin solicitudes». Si el archivo no existe, termina con un error y código de salida 1.

## 4. Versiones guardadas y versión activa

| Versión | Configuración | RMSE (prueba) | Estado |
|---|---|---|---|
| `v1` | Random Forest, 200 árboles, sin límite de profundidad, semilla 42 | 1.1855 | Versión anterior |
| `v2` | Random Forest, 300 árboles, `max_depth=12`, semilla 42 | 1.1773 | **Activa** |

Ambas están en `models/v1/` y `models/v2/` (modelo, métricas y metadatos) y se listan con `GET /models`. Una tercera versión generada con `POST /train` aparece como `models/v3/` (ver Fase 3) y se puede usar igual.

La API usa la versión indicada por la variable de entorno **`MODEL_VERSION`**, cuyo valor por defecto en esta fase es `v2` (la de la Fase 3 es `v1`). **Para volver a la anterior** basta reiniciar con la configuración existente:

```
$env:MODEL_VERSION="v1"; python -m flask --app api run --host 0.0.0.0 --port 8000      # PowerShell
MODEL_VERSION=v1 python -m flask --app api run --host 0.0.0.0 --port 8000              # Bash
docker run ... -e MODEL_VERSION=v1 ...                                                 # Docker
```

La columna `version` del registro permite comprobar con cuál versión se respondió cada solicitud.

## 5. Política de reentrenamiento

Se revisará el modelo al reunir **500 pedidos nuevos con su precio de venta real**, o cuando la carta cambie (pizzas, tamaños o ingredientes nuevos) o aparezcan rechazos 422 recurrentes por valores desconocidos; un campo ausente, en cambio, se corrige en quien envía los datos. El responsable del modelo (integrante designado por el equipo) añadirá esos pedidos al dataset y ejecutará `POST /train` para obtener una versión candidata con la misma semilla y partición por tipo de pizza de la Fase 2. Comparará el **RMSE** (métrica principal de la Fase 2, en `metrics.json`) de la candidata y de la versión activa sobre los mismos casos reservados y, solo si el RMSE mejora, la activará reiniciando con `MODEL_VERSION=<candidata>`; si no mejora, se conserva la versión activa.

_Limitación:_ las métricas de `metrics.json` solo son comparables si ambas versiones se evalúan sobre los mismos tipos de pizza reservados; si el dataset cambia mucho, debe fijarse ese conjunto de prueba. El porcentaje de fallos mide el funcionamiento del servicio, **no** la calidad de las predicciones (eso requiere conocer el precio real).

## 6. Reporte

**Solicitudes enviadas** (API con `MODEL_VERSION=v2`, registro vacío, `python client.py --filas 2 --incluir-invalida`):

| # | Solicitud a `POST /predict` | Respuesta |
|---|---|---|
| 1 | `{"pizza_size": "S", "pizza_category": "Classic", "pizza_ingredients": "Sliced Ham, Pineapple, Mozzarella Cheese"}` | HTTP 200 `{"prediccion": 10.5, "version": "v2"}` |
| 2 | `{"pizza_size": "M", "pizza_category": "Classic", "pizza_ingredients": "Pepperoni, Mushrooms, Red Onions, Red Peppers, Bacon"}` | HTTP 200 `{"prediccion": 16.0, "version": "v2"}` |
| 3 | `{"pizza_size": "S", "pizza_category": "Classic"}` (falta `pizza_ingredients`) | HTTP 422 `Falta el campo obligatorio 'pizza_ingredients'.` |

**Indicadores** obtenidos con `python monitor.py --log logs/predict_log.csv`:

```
Total de solicitudes: 3
Solicitudes fallidas (HTTP 4xx/5xx): 1
Porcentaje de solicitudes fallidas: 33,3 %
Solicitudes por versión: (sin modelo): 1, v2: 2
```

**Interpretación:** la API procesó dos registros con `v2` y rechazó el tercero porque faltaba `pizza_ingredients`; la tasa de fallos fue 1/3 (33,3 %), el resultado esperado de la prueba de validación (cada intento generó una sola fila, también el rechazado). Los 3 eventos están en `docs/muestra_registro.csv`.

**Versiones:** activa `v2`, anterior `v1` (sección 4). **Política:** sección 5.

_Estos valores provienen de la ejecución de validación del paquete; al repetir el ensayo en su equipo, actualice aquí fecha, horas y resultados con los suyos._

## 7. Pruebas

`python -m pytest` ejecuta las pruebas de las fases anteriores y `tests/test_monitoreo.py`: una fila por solicitud (incluida la rechazada), historial conservado entre ejecuciones con un solo encabezado, registro de 400/422/503, endpoints no registrados, fallo del registro sin romper la API, el caso «2 válidas + 1 inválida = 3 solicitudes y 33,3 %», registro vacío («sin solicitudes»), clasificación de 4xx/5xx como fallidas, archivo inexistente (código ≠ 0) y versión activa por defecto `v2` con regreso a `v1`.

## 8. Alcance

Se cubre lo requerido para el cierre: demostración local en Docker, pruebas y reporte. No se implementaron el despliegue en la nube, tableros gráficos ni reentrenamiento automático (opcionales). La API sigue lista para un futuro frontend (CORS, `/schema`, `/models`).
