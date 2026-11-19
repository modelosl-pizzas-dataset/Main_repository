"""Almacenamiento y carga de versiones del modelo (carpetas models/<version>/)."""
import json
import re
import shutil
import tempfile
from pathlib import Path

import joblib

from src.errors import ModelNotFoundError, VersionExistsError

MODEL_FILE = "modelo.joblib"
METRICS_FILE = "metrics.json"
METADATA_FILE = "metadata.json"
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_version_id(version):
    if not _VERSION_RE.match(version or ""):
        raise VersionExistsError(
            f"Identificador de versión no válido: '{version}'. Use letras, números, '.', '_' o '-'."
        )


def ensure_version_free(output_dir, version):
    """Lanza VersionExistsError si la versión ya existe (se usa antes de entrenar y al guardar)."""
    validate_version_id(version)
    final_dir = Path(output_dir) / version
    if final_dir.exists():
        raise VersionExistsError(
            f"La versión '{version}' ya existe en {final_dir}. Elija otro identificador (no se sobrescribe)."
        )
    return final_dir


def save_version(output_dir, version, bundle, metrics, metadata):
    """Guarda modelo + métricas + metadatos. NUNCA sobrescribe una versión existente.

    Se escribe primero en una carpeta temporal y luego se renombra, de modo que una
    ejecución fallida no deja versiones a medias.
    """
    final_dir = ensure_version_free(output_dir, version)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".tmp_{version}_", dir=output_dir))
    try:
        joblib.dump(bundle, tmp / MODEL_FILE)
        (tmp / METRICS_FILE).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        (tmp / METADATA_FILE).write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.chmod(0o755)  # mkdtemp crea 0700; evita problemas de lectura con volúmenes de Docker
        tmp.rename(final_dir)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return {
        "model": final_dir / MODEL_FILE,
        "metrics": final_dir / METRICS_FILE,
        "metadata": final_dir / METADATA_FILE,
    }


def resolve_model_path(model):
    """Acepta la ruta al .joblib o a la carpeta de la versión."""
    path = Path(model)
    if path.is_dir():
        path = path / MODEL_FILE
    if not path.is_file():
        raise ModelNotFoundError(f"No se encontró el modelo: {model}")
    return path


def load_version(model):
    """Carga el bundle guardado (pipeline completo + esquema de entrada)."""
    path = resolve_model_path(model)
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        raise ModelNotFoundError(f"No se pudo leer el modelo '{path}': {exc}") from exc
    if not isinstance(bundle, dict) or "pipeline" not in bundle or "input_columns" not in bundle:
        raise ModelNotFoundError(f"El archivo '{path}' no es un modelo válido de esta aplicación.")
    return bundle
