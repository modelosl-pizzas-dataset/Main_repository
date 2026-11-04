"""Entrenamiento y evaluación (reutilizable por train.py y, en la Fase 3, por la API /train)."""
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline

from src.config import DEFAULT_RF_PARAMS, GROUP_COLUMN, INPUT_SCHEMA, NULLABLE_COLUMNS, SIZE_ORDER
from src.data import file_sha256, read_csv, validate_training_frame
from src.errors import AppError
from src.features import build_preprocessor, to_model_frame
from src.versioning import ensure_version_free, save_version


def build_pipeline(estimator):
    return Pipeline([("preprocesamiento", build_preprocessor()), ("modelo", estimator)])


def split_data(df, test_size, seed, group_column=None):
    """Separa train/test ANTES de ajustar transformaciones. Con group_column, un mismo grupo
    (tipo de pizza) nunca queda en ambos conjuntos."""
    idx = np.arange(len(df))
    if group_column:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(idx, groups=df[group_column].to_numpy()))
    else:
        train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed)
    return train_idx, test_idx


def compute_metrics(y_true, y_pred):
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def get_git_commit(explicit=None):
    if explicit:
        return explicit
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                             cwd=Path(__file__).resolve().parent.parent)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "no-disponible"


def train(data_path, target, output_dir, version, seed=42, test_size=0.2,
          group_column=GROUP_COLUMN, n_estimators=None, max_depth=None, git_commit=None, verbose=True):
    """Ejecuta el proceso completo y guarda una versión. Devuelve metrics, metadata y rutas."""
    ensure_version_free(output_dir, version)  # falla rápido, antes de entrenar
    if not 0 < test_size < 1:
        raise AppError("--test-size debe estar entre 0 y 1.")

    df = read_csv(data_path, "archivo de datos")
    validate_training_frame(df, target=target, group_column=group_column)

    train_idx, test_idx = split_data(df, test_size, seed, group_column)
    if group_column:
        overlap = set(df[group_column].iloc[train_idx]) & set(df[group_column].iloc[test_idx])
        assert not overlap, "Fuga: grupos repetidos entre entrenamiento y prueba"

    input_columns = list(INPUT_SCHEMA)
    X = to_model_frame(df, input_columns)
    y = df[target].to_numpy(dtype=float)
    X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]

    params = dict(DEFAULT_RF_PARAMS)
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    if max_depth is not None:
        params["max_depth"] = max_depth
    params.update({"random_state": seed, "n_jobs": -1})

    # Modelo final (Random Forest) y modelo base (regresión lineal) con la MISMA partición y métrica
    modelo = build_pipeline(RandomForestRegressor(**params)).fit(X_train, y_train)
    base = build_pipeline(LinearRegression()).fit(X_train, y_train)
    referencia = DummyRegressor(strategy="mean").fit(X_train, y_train)

    metrics = {
        "metrica_principal": "rmse",
        "modelo": {"algoritmo": "RandomForestRegressor", **compute_metrics(y_test, modelo.predict(X_test))},
        "modelo_base": {"algoritmo": "LinearRegression", **compute_metrics(y_test, base.predict(X_test))},
        "referencia_media": {"algoritmo": "DummyRegressor(mean)", **compute_metrics(y_test, referencia.predict(X_test))},
        "entrenamiento_rmse": float(root_mean_squared_error(y_train, modelo.predict(X_train))),
    }

    n_features = len(modelo.named_steps["preprocesamiento"].get_feature_names_out())
    bundle = {
        "pipeline": modelo,
        "input_columns": input_columns,
        "input_schema": INPUT_SCHEMA,
        "nullable_columns": NULLABLE_COLUMNS,
        "allowed_sizes": SIZE_ORDER,
        "target": target,
        "version": version,
    }
    metadata = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "problem_type": "regresion",
        "target": target,
        "algorithm": "RandomForestRegressor (sklearn Pipeline)",
        "input_features": INPUT_SCHEMA,
        "nullable_features": NULLABLE_COLUMNS,
        "n_features_after_preprocessing": n_features,
        "model_params": {k: v for k, v in params.items()},
        "seed": seed,
        "data": {
            "source": "Pizza Sales (data/pizza_sales.csv, dataset de Fase 1)",
            "file": str(data_path),
            "sha256": file_sha256(data_path),
            "n_observations": int(len(df)),
        },
        "split": {
            "strategy": f"GroupShuffleSplit por '{group_column}'" if group_column else "train_test_split aleatorio",
            "group_column": group_column,
            "test_size": test_size,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_groups_train": int(df[group_column].iloc[train_idx].nunique()) if group_column else None,
            "n_groups_test": int(df[group_column].iloc[test_idx].nunique()) if group_column else None,
        },
        "metric": "RMSE (principal), MAE, R2",
        "python_version": platform.python_version(),
        "library_versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__},
        "requirements_file": "requirements.txt",
        "git_commit": get_git_commit(git_commit),
    }
    paths = save_version(output_dir, version, bundle, metrics, metadata)
    return {"metrics": metrics, "metadata": metadata, "paths": paths, "bundle": bundle}
