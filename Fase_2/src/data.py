"""Lectura, validación y preparación de datos (sin entrenar nada)."""
import hashlib
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.config import GROUP_COLUMN, INPUT_SCHEMA, SIZE_ORDER, TARGET
from src.errors import DataValidationError


def read_csv(path, what="archivo"):
    """Lee un CSV con errores claros."""
    path = Path(path)
    if not path.is_file():
        raise DataValidationError(f"No se encontró el {what}: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # CSV corrupto, vacío, etc.
        raise DataValidationError(f"No se pudo leer el {what} '{path}': {exc}") from exc
    if df.empty:
        raise DataValidationError(f"El {what} '{path}' no contiene filas.")
    return df


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_dataset(raw_df):
    """Transformación SIN estado (no aprende nada de los datos): selecciona columnas útiles.

    Se descartan: identificadores, fechas/horas, nombre comercial redundante, quantity
    (no influye en el precio unitario) y total_price (= unit_price x quantity: fuga de información).
    Se conserva pizza_name únicamente como GRUPO para separar train/test.
    """
    needed = [GROUP_COLUMN, "pizza_size", "pizza_category", "pizza_ingredients", TARGET]
    missing = [c for c in needed if c not in raw_df.columns]
    if missing:
        raise DataValidationError(f"Al CSV crudo le faltan columnas: {missing}")
    out = raw_df[needed].copy()
    for col in [GROUP_COLUMN, "pizza_size", "pizza_category", "pizza_ingredients"]:
        out[col] = out[col].astype("string").str.strip()
    return out.reset_index(drop=True)


def validate_input_frame(df, required=None, allowed_sizes=SIZE_ORDER):
    """Valida el CSV de predicción: columnas requeridas, tipos y valores de tamaño.

    - Falta una columna  -> error.   - Celda nula -> permitido (se imputa).
    - Columnas extra -> se ignoran.   - Tipo numérico en columna de texto -> error.
    - Tamaño desconocido -> error.    - Categoría/ingrediente no visto -> se ignora en la transformación.
    """
    required = list(required or INPUT_SCHEMA)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"Faltan columnas requeridas: {missing}. Columnas recibidas: {list(df.columns)}"
        )
    for col in required:
        if is_numeric_dtype(df[col]) and df[col].notna().any():
            raise DataValidationError(
                f"La columna '{col}' debe contener texto, pero se recibió un tipo numérico ({df[col].dtype})."
            )
    if "pizza_size" in required:
        sizes = df["pizza_size"].dropna().astype(str).str.strip()
        bad = sorted(set(sizes[~sizes.isin(allowed_sizes)]))
        if bad:
            raise DataValidationError(f"Valores de 'pizza_size' no válidos: {bad}. Permitidos: {allowed_sizes}")
    if df.empty:
        raise DataValidationError("El archivo de entrada no contiene filas.")
    return [c for c in df.columns if c not in required]  # columnas extra (ignoradas)


def validate_training_frame(df, target=TARGET, group_column=None):
    """Valida el dataset etiquetado antes de entrenar."""
    required = list(INPUT_SCHEMA) + [target] + ([group_column] if group_column else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"Faltan columnas en los datos de entrenamiento: {missing}")
    if not is_numeric_dtype(df[target]):
        raise DataValidationError(f"La variable objetivo '{target}' debe ser numérica.")
    if df[target].isna().any():
        raise DataValidationError(f"La variable objetivo '{target}' contiene nulos.")
    validate_input_frame(df, required=list(INPUT_SCHEMA))
