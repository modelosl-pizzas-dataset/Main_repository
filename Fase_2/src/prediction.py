"""Predicción con una versión guardada: no reentrena ni reajusta transformaciones."""
import os
import tempfile
from pathlib import Path

import numpy as np

from src.data import read_csv, validate_input_frame
from src.errors import AppError
from src.features import to_model_frame
from src.versioning import load_version

PREDICTION_COLUMN = "prediccion"


def predict_dataframe(bundle, df):
    """Devuelve (df_resultado, columnas_extra_ignoradas). Una predicción por fila, mismo orden."""
    extra = validate_input_frame(df, required=bundle["input_columns"], allowed_sizes=bundle["allowed_sizes"])
    X = to_model_frame(df, bundle["input_columns"])  # reordena según el esquema de entrenamiento
    preds = bundle["pipeline"].predict(X)
    if len(preds) != len(df) or not np.isfinite(preds).all():
        raise AppError("La predicción produjo un número de filas distinto o valores no finitos.")
    result = df.copy()
    result[PREDICTION_COLUMN] = preds
    return result, extra


def predict_file(model, input_path, output_path):
    bundle = load_version(model)
    df = read_csv(input_path, "archivo de entrada")
    result, extra = predict_dataframe(bundle, df)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Escritura atómica: si algo falla no queda un CSV que aparente ser válido
    fd, tmp = tempfile.mkstemp(suffix=".csv", dir=output_path.parent)
    os.close(fd)
    try:
        result.to_csv(tmp, index=False)
        os.replace(tmp, output_path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise
    return {"n_rows": len(result), "extra_columns": extra, "version": bundle.get("version")}
