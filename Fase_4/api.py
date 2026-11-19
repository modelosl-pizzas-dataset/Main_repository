"""API REST (Flask) del modelo de precio unitario de pizzas.

Rutas:  GET /health · POST /predict · POST /train · GET /models · GET /schema · GET /

Arranque local (desde esta carpeta):
    python -m flask --app api run --host 0.0.0.0 --port 8000
La versión del modelo se elige con la variable de entorno MODEL_VERSION y el registro persistente de
/predict se guarda en LOG_PATH (ver src/settings.py).
"""
import threading
from pathlib import Path

from flask import Flask, g, jsonify, request
from werkzeug.exceptions import HTTPException

from src.errors import AppError, DataValidationError, ModelNotFoundError, VersionExistsError
from src.request_log import RequestLogger
from src.service import (ModelStore, RecordValidationError, input_schema, list_versions, next_version_id,
                         predict_record, validate_record)
from src.settings import Settings
from src.training import train
from src.versioning import validate_version_id

BASE_DIR = Path(__file__).resolve().parent


def _error(status, message, tipo, campos=None):
    body = {"error": message, "tipo": tipo}
    if campos:
        body["campos"] = campos
    return jsonify(body), status


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _parse_train_body(payload, settings):
    """Valida el cuerpo (opcional) de /train. Devuelve los parámetros o lanza RecordValidationError."""
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise RecordValidationError("El cuerpo de /train debe ser un objeto JSON (o estar vacío).")
    allowed = {"version", "seed", "n_estimators", "max_depth", "test_size"}
    unknown = [k for k in payload if k not in allowed]
    if unknown:
        raise RecordValidationError(
            f"Campos no reconocidos en /train: {', '.join(unknown)}. Permitidos: {', '.join(sorted(allowed))}.", unknown)

    def check(name, ok, expected):
        if name in payload and payload[name] is not None and not ok(payload[name]):
            raise RecordValidationError(f"El campo '{name}' debe ser {expected}.", [name])

    check("version", lambda v: isinstance(v, str) and v.strip(), "texto no vacío")
    check("seed", _is_int, "un entero")
    check("n_estimators", lambda v: _is_int(v) and v > 0, "un entero positivo")
    check("max_depth", lambda v: _is_int(v) and v > 0, "un entero positivo o null")
    check("test_size", lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and 0 < v < 1,
          "un número entre 0 y 1")
    return {
        "version": (payload.get("version") or "").strip() or None,
        "seed": payload.get("seed", settings.seed),
        "n_estimators": payload.get("n_estimators"),
        "max_depth": payload.get("max_depth"),
        "test_size": payload.get("test_size", settings.test_size),
    }


def create_app(settings=None):
    settings = settings or Settings.from_env(BASE_DIR)
    app = Flask(__name__)
    app.json.ensure_ascii = False
    app.json.sort_keys = False
    store = ModelStore(settings)
    request_logger = RequestLogger(settings.log_path) if settings.log_path else None
    train_lock = threading.Lock()
    app.config["SETTINGS"] = settings
    app.extensions["model_store"] = store

    # ---------- CORS (para un futuro frontend; desactivar con CORS_ORIGINS="") ----------
    @app.after_request
    def add_cors(response):
        if settings.cors_origins:
            response.headers["Access-Control-Allow-Origin"] = settings.cors_origins
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    # ---------- Registro de solicitudes (Fase 4) ----------
    # after_request se ejecuta UNA vez por solicitud, también cuando la entrada es rechazada o hay un error,
    # por lo que cada intento de POST /predict genera exactamente una fila.
    @app.after_request
    def log_predict_request(response):
        if request_logger is not None and request.method == "POST" and request.path == "/predict":
            exito = 200 <= response.status_code < 300
            try:
                request_logger.log(response.status_code,
                                   g.get("log_version") if exito else None,
                                   g.get("log_prediction") if exito else None)
            except Exception:  # un fallo del registro nunca debe romper la API
                app.logger.exception("No se pudo escribir el registro de solicitudes")
        return response

    # ---------- Errores siempre en JSON ----------
    @app.errorhandler(HTTPException)
    def handle_http(exc):
        mensajes = {404: "La ruta solicitada no existe.", 405: "Método no permitido para esta ruta.",
                    413: "El cuerpo de la solicitud es demasiado grande."}
        return _error(exc.code, mensajes.get(exc.code, exc.description), "http")

    @app.errorhandler(Exception)
    def handle_unexpected(exc):
        app.logger.exception("Error no previsto")
        return _error(500, f"Error interno del servidor: {type(exc).__name__}", "interno")

    # ---------- Rutas ----------
    @app.get("/")
    def index():
        return jsonify({
            "servicio": "API de predicción de precio unitario de pizzas",
            "version_activa": settings.model_version,
            "endpoints": {
                "GET /health": "estado de la API",
                "POST /predict": "predicción para un registro JSON",
                "POST /train": "entrena y guarda una nueva versión",
                "GET /models": "versiones guardadas y versión activa",
                "GET /schema": "variables de entrada esperadas",
            },
        })

    @app.get("/health")
    def health():
        try:
            store.load_active()
            disponible = True
        except AppError:
            disponible = False
        return jsonify({"status": "ok", "servicio": "api en ejecución",
                        "modelo_activo": settings.model_version, "modelo_disponible": disponible})

    @app.post("/predict")
    def predict():
        payload = request.get_json(force=True, silent=True)
        if payload is None:
            return _error(400, "El cuerpo debe ser un JSON válido: un objeto con las variables predictoras.", "json_invalido")
        try:
            # 1) validar el registro (no requiere el modelo) -> 422 con el nombre del campo
            validate_record(payload)
            # 2) cargar la versión activa -> 503 si no está disponible
            version, bundle = store.load_active()
            # 3) mismas transformaciones y modelo guardados
            result, ignored = predict_record(bundle, payload)
        except RecordValidationError as exc:
            return _error(422, str(exc), "validacion", exc.campos)
        except ModelNotFoundError as exc:
            return _error(503, str(exc), "modelo_no_disponible")
        except DataValidationError as exc:
            return _error(422, str(exc), "validacion")
        result["version"] = version
        g.log_version, g.log_prediction = version, result["prediccion"]
        if ignored:
            result["campos_ignorados"] = ignored
        return jsonify(result), 200

    @app.post("/train")
    def train_endpoint():
        payload = None
        if request.get_data(cache=True).strip():
            payload = request.get_json(force=True, silent=True)
            if payload is None:
                return _error(400, "El cuerpo debe ser un JSON válido o estar vacío.", "json_invalido")
        try:
            params = _parse_train_body(payload, settings)
        except RecordValidationError as exc:
            return _error(422, str(exc), "validacion", exc.campos)

        if not train_lock.acquire(blocking=False):
            return _error(409, "Ya hay un entrenamiento en curso; intente de nuevo cuando termine.", "entrenamiento_en_curso")
        try:
            version = params["version"] or next_version_id(settings.models_dir)
            try:
                validate_version_id(version)
            except VersionExistsError as exc:  # identificador con caracteres no válidos
                return _error(422, str(exc), "validacion", ["version"])
            if (settings.models_dir / version).exists():
                return _error(409, f"La versión '{version}' ya existe; no se sobrescribe. Elija otro identificador "
                                   "u omita 'version' para generar uno disponible.", "version_existente", ["version"])
            try:
                res = train(str(settings.data_path), settings.target, str(settings.models_dir), version,
                            seed=params["seed"], test_size=params["test_size"], group_column=settings.group_column,
                            n_estimators=params["n_estimators"], max_depth=params["max_depth"])
            except VersionExistsError as exc:
                return _error(409, str(exc), "version_existente", ["version"])
            except AppError as exc:  # datos de entrenamiento ausentes o inválidos: problema de configuración
                return _error(500, f"No se pudo entrenar: {exc}", "configuracion")
        finally:
            train_lock.release()

        folder = settings.models_dir.name
        return jsonify({
            "version": version,
            "mensaje": "Versión entrenada y guardada. Las versiones anteriores se conservaron.",
            "metricas": {k: res["metrics"][k] for k in ("modelo", "modelo_base", "referencia_media")},
            "archivos": {"modelo": f"{folder}/{version}/modelo.joblib",
                         "metricas": f"{folder}/{version}/metrics.json",
                         "metadatos": f"{folder}/{version}/metadata.json"},
            "version_activa": settings.model_version,
            "nota": "La API sigue prediciendo con la versión activa. Para usar la nueva, "
                    f"reinicie con MODEL_VERSION={version}.",
        }), 201

    @app.get("/models")
    def models():
        available = store.available_versions()
        return jsonify({"version_activa": settings.model_version,
                        "versiones": list_versions(settings.models_dir, settings.model_version, available)})

    @app.get("/schema")
    def schema():
        try:
            _, bundle = store.load_active()
        except ModelNotFoundError as exc:
            return _error(503, str(exc), "modelo_no_disponible")
        return jsonify(input_schema(bundle))

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
