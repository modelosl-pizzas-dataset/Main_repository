"""Lógica de servicio reutilizada por la API: validación de registros JSON, carga del modelo activo,
predicción de un registro y gestión de versiones. No depende de Flask."""
import json
import re
import threading

import pandas as pd

from src.config import INPUT_SCHEMA, SIZE_ORDER
from src.errors import DataValidationError, ModelNotFoundError
from src.prediction import PREDICTION_COLUMN, predict_dataframe
from src.versioning import METADATA_FILE, METRICS_FILE, MODEL_FILE, load_version


class RecordValidationError(DataValidationError):
    """El JSON recibido no cumple el esquema; `campos` identifica las variables implicadas."""

    def __init__(self, message, campos=None):
        super().__init__(message)
        self.campos = list(campos or [])


def _type_name(value):
    if isinstance(value, bool):
        return "booleano"
    if isinstance(value, (int, float)):
        return "número"
    if isinstance(value, list):
        return "lista"
    if isinstance(value, dict):
        return "objeto"
    return type(value).__name__


def validate_record(payload, required=None):
    """Valida un registro JSON. Devuelve (registro, campos_ignorados).

    - Debe ser un objeto JSON con TODAS las variables requeridas (aunque su valor sea null).
    - Cada valor debe ser texto o null (null/cadena vacía = dato faltante; se imputa con la regla guardada).
    - Claves adicionales se ignoran y se informan.
    """
    required = list(required or INPUT_SCHEMA)
    if not isinstance(payload, dict):
        raise RecordValidationError("El cuerpo debe ser un objeto JSON con las variables predictoras: "
                                    + ", ".join(f"'{c}'" for c in required) + ".")
    missing = [c for c in required if c not in payload]
    if missing:
        names = ", ".join(f"'{c}'" for c in missing)
        raise RecordValidationError(
            f"Falta el campo obligatorio {names}." if len(missing) == 1 else f"Faltan los campos obligatorios: {names}.",
            missing)
    record = {}
    for col in required:
        value = payload[col]
        if value is not None and not isinstance(value, str):
            raise RecordValidationError(
                f"El campo '{col}' debe ser texto (string) o null; se recibió {_type_name(value)} ({value!r}).", [col])
        if isinstance(value, str):
            value = value.strip() or None
        record[col] = value
    ignored = [k for k in payload if k not in required]
    return record, ignored


def predict_record(bundle, payload):
    """Una predicción para un registro JSON, con el MISMO preprocesamiento y modelo guardados."""
    record, ignored = validate_record(payload, bundle["input_columns"])
    df = pd.DataFrame([record], columns=bundle["input_columns"], dtype=object)  # columnas en el orden del esquema
    try:
        result, _ = predict_dataframe(bundle, df)
    except DataValidationError as exc:  # p. ej. tamaño no permitido
        campos = ["pizza_size"] if "pizza_size" in str(exc) else []
        raise RecordValidationError(str(exc), campos) from exc
    out = {"prediccion": float(result[PREDICTION_COLUMN].iloc[0])}
    return out, ignored


def _natural_key(name):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


class ModelStore:
    """Localiza y cachea la versión activa; detecta si el archivo cambió."""

    def __init__(self, settings):
        self.settings = settings
        self._cache = {}
        self._lock = threading.Lock()

    def available_versions(self):
        root = self.settings.models_dir
        if not root.is_dir():
            return []
        return sorted((p.name for p in root.iterdir() if (p / MODEL_FILE).is_file()), key=_natural_key)

    def load_active(self):
        """Devuelve (version, bundle) o lanza ModelNotFoundError con instrucciones."""
        version = self.settings.model_version
        path = self.settings.models_dir / version / MODEL_FILE
        if not path.is_file():
            disponibles = self.available_versions()
            raise ModelNotFoundError(
                f"El modelo configurado '{version}' no está disponible en '{self.settings.models_dir}'. "
                f"Versiones disponibles: {', '.join(disponibles) if disponibles else 'ninguna'}. "
                "Seleccione una con la variable de entorno MODEL_VERSION y reinicie la API.")
        stamp = path.stat().st_mtime_ns
        with self._lock:
            cached = self._cache.get(path)
            if cached and cached[0] == stamp:
                return version, cached[1]
            bundle = load_version(path)
            self._cache[path] = (stamp, bundle)
            return version, bundle


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def list_versions(models_dir, active, available):
    """Resumen de las versiones guardadas (para GET /models)."""
    items = []
    for version in available:
        meta = _read_json(models_dir / version / METADATA_FILE)
        metrics = _read_json(models_dir / version / METRICS_FILE).get("modelo", {})
        items.append({
            "version": version,
            "activa": version == active,
            "creada": meta.get("created_at"),
            "algoritmo": meta.get("algorithm"),
            "rmse": metrics.get("rmse"),
            "mae": metrics.get("mae"),
            "r2": metrics.get("r2"),
        })
    return items


def next_version_id(models_dir):
    """Siguiente identificador libre del estilo v<N> (v3 si existen v1 y v2)."""
    numbers = [int(m.group(1)) for p in (models_dir.iterdir() if models_dir.is_dir() else [])
               if (m := re.fullmatch(r"v(\d+)", p.name))]
    n = max(numbers, default=0) + 1
    while (models_dir / f"v{n}").exists():
        n += 1
    return f"v{n}"


def input_schema(bundle):
    """Descripción del esquema de entrada (para que un futuro frontend construya formularios)."""
    variables = []
    pre = bundle["pipeline"].named_steps["preprocesamiento"]
    for col in bundle["input_columns"]:
        info = {"nombre": col, "tipo": "texto", "admite_nulo": col in bundle.get("nullable_columns", [])}
        if col == "pizza_size":
            info["valores_permitidos"] = list(bundle.get("allowed_sizes", SIZE_ORDER))
        try:
            if col == "pizza_category":
                encoder = pre.named_transformers_["categoria"].named_steps["onehot"]
                info["valores_conocidos"] = [str(v) for v in encoder.categories_[0]]
            if col == "pizza_ingredients":
                vec = pre.named_transformers_["ingredientes"].named_steps["vectorizer"]
                info["formato"] = "ingredientes separados por coma"
                info["ingredientes_conocidos"] = list(vec.vocabulary_)
        except (KeyError, AttributeError):
            pass
        variables.append(info)
    return {"variables": variables, "variable_objetivo": bundle.get("target"),
            "nota": "Las variables no listadas se ignoran; null o cadena vacía = dato faltante (se imputa)."}
