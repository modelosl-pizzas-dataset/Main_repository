"""Pruebas básicas de la Fase 2. Usan datos sintéticos y directorios temporales (sin red)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.errors import DataValidationError, VersionExistsError
from src.prediction import predict_file
from src.training import train
from src.versioning import load_version

ROOT = Path(__file__).resolve().parent.parent
FAST = dict(seed=42, n_estimators=10, verbose=False)


def entrenar(csv, out_dir, version, **kw):
    return train(str(csv), "unit_price", str(out_dir), version, **{**FAST, **kw})


# ---------------------------------------------------------------- 1. Entrenamiento y almacenamiento
def test_entrenamiento_genera_artefacto_metricas_y_metadatos(synthetic_csv, tmp_path):
    res = entrenar(synthetic_csv, tmp_path / "models", "v1")
    for key in ("model", "metrics", "metadata"):
        assert res["paths"][key].is_file()

    bundle = load_version(res["paths"]["model"])
    assert bundle["input_columns"] == ["pizza_size", "pizza_category", "pizza_ingredients"]

    metrics = json.loads(res["paths"]["metrics"].read_text(encoding="utf-8"))
    for section in ("modelo", "modelo_base"):
        assert {"rmse", "mae", "r2"} <= set(metrics[section])
        assert metrics[section]["rmse"] >= 0

    meta = json.loads(res["paths"]["metadata"].read_text(encoding="utf-8"))
    for key in ("version", "created_at", "target", "algorithm", "input_features", "seed", "data",
                "split", "python_version", "git_commit"):
        assert key in meta
    assert meta["version"] == "v1" and meta["target"] == "unit_price"
    assert meta["data"]["n_observations"] == 360


def test_separacion_por_grupos_sin_fuga(synthetic_csv, tmp_path):
    res = entrenar(synthetic_csv, tmp_path / "models", "v1")
    split = res["metadata"]["split"]
    assert split["n_train"] + split["n_test"] == 360
    assert split["n_groups_train"] + split["n_groups_test"] == 8  # cada tipo de pizza en un solo conjunto


# ---------------------------------------------------------------- 2. Carga y predicción
def test_prediccion_conserva_resultados_del_modelo_guardado(synthetic_csv, new_data, tmp_path):
    res = entrenar(synthetic_csv, tmp_path / "models", "v1")
    entrada, salida = tmp_path / "nuevos.csv", tmp_path / "out" / "pred.csv"
    new_data.to_csv(entrada, index=False)

    info = predict_file(res["paths"]["model"], entrada, salida)
    pred = pd.read_csv(salida)

    assert info["n_rows"] == len(new_data) == len(pred)
    assert list(pred["pizza_size"]) == list(new_data["pizza_size"])  # mismo orden de filas
    esperado = res["bundle"]["pipeline"].predict(new_data[res["bundle"]["input_columns"]])
    np.testing.assert_allclose(pred["prediccion"].to_numpy(), esperado, rtol=1e-9)


def test_prediccion_no_depende_del_orden_de_columnas_ni_de_columnas_extra(synthetic_csv, new_data, tmp_path):
    res = entrenar(synthetic_csv, tmp_path / "models", "v1")
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    new_data.to_csv(a, index=False)
    desordenado = new_data[["pizza_ingredients", "pizza_size", "pizza_category"]].assign(extra=1)
    desordenado.to_csv(b, index=False)
    predict_file(res["paths"]["model"], a, tmp_path / "pa.csv")
    info = predict_file(res["paths"]["model"], b, tmp_path / "pb.csv")
    assert info["extra_columns"] == ["extra"]
    np.testing.assert_allclose(pd.read_csv(tmp_path / "pa.csv")["prediccion"],
                               pd.read_csv(tmp_path / "pb.csv")["prediccion"])


# ---------------------------------------------------------------- 3. Valores faltantes
def test_valores_faltantes_se_imputan_sin_eliminar_filas(synthetic_csv, tmp_path):
    res = entrenar(synthetic_csv, tmp_path / "models", "v1")
    nuevos = pd.DataFrame({
        "pizza_size": ["M", None, "L", "S"],
        "pizza_category": ["Classic", "Veggie", None, "Chicken"],
        "pizza_ingredients": ["Ham, Cheese", "Mushrooms", "Chicken", None],
    })
    entrada, salida = tmp_path / "nulos.csv", tmp_path / "pred.csv"
    nuevos.to_csv(entrada, index=False)
    predict_file(res["paths"]["model"], entrada, salida)
    pred = pd.read_csv(salida)
    assert len(pred) == 4                       # no se eliminan observaciones
    assert pred["prediccion"].notna().all()     # todas tienen predicción
    assert np.isfinite(pred["prediccion"]).all()


def test_categoria_e_ingrediente_no_vistos_se_ignoran(synthetic_csv, tmp_path):
    res = entrenar(synthetic_csv, tmp_path / "models", "v1")
    nuevos = pd.DataFrame({"pizza_size": ["M"], "pizza_category": ["Postre"],
                           "pizza_ingredients": ["Ham, IngredienteDesconocido"]})
    entrada = tmp_path / "x.csv"
    nuevos.to_csv(entrada, index=False)
    predict_file(res["paths"]["model"], entrada, tmp_path / "p.csv")
    assert len(pd.read_csv(tmp_path / "p.csv")) == 1


# ---------------------------------------------------------------- 4. Entrada inválida
def test_columna_requerida_ausente_da_error_claro(synthetic_csv, new_data, tmp_path):
    res = entrenar(synthetic_csv, tmp_path / "models", "v1")
    entrada = tmp_path / "incompleto.csv"
    new_data.drop(columns=["pizza_ingredients"]).to_csv(entrada, index=False)
    with pytest.raises(DataValidationError, match="pizza_ingredients"):
        predict_file(res["paths"]["model"], entrada, tmp_path / "no_debe_existir.csv")
    assert not (tmp_path / "no_debe_existir.csv").exists()


def test_cli_predict_con_error_termina_con_codigo_distinto_de_cero(synthetic_csv, new_data, tmp_path):
    res = entrenar(synthetic_csv, tmp_path / "models", "v1")
    entrada, salida = tmp_path / "malo.csv", tmp_path / "salida.csv"
    new_data.drop(columns=["pizza_size"]).to_csv(entrada, index=False)
    cmd = [sys.executable, "predict.py", "--model", str(res["paths"]["model"]),
           "--input", str(entrada), "--output", str(salida)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0
    assert "pizza_size" in r.stderr
    assert not salida.exists()

    r2 = subprocess.run([sys.executable, "predict.py", "--model", str(tmp_path / "nada.joblib"),
                         "--input", str(entrada), "--output", str(salida)], cwd=ROOT, capture_output=True, text=True)
    assert r2.returncode != 0 and not salida.exists()


def test_tipo_o_tamano_incompatible_da_error(synthetic_csv, new_data, tmp_path):
    res = entrenar(synthetic_csv, tmp_path / "models", "v1")
    numerico = new_data.assign(pizza_size=[1, 2, 3])
    with pytest.raises(DataValidationError, match="texto"):
        entrada = tmp_path / "n.csv"
        numerico.to_csv(entrada, index=False)
        predict_file(res["paths"]["model"], entrada, tmp_path / "o.csv")
    invalido = new_data.assign(pizza_size=["S", "XS", "L"])
    entrada2 = tmp_path / "t.csv"
    invalido.to_csv(entrada2, index=False)
    with pytest.raises(DataValidationError, match="XS"):
        predict_file(res["paths"]["model"], entrada2, tmp_path / "o.csv")


# ---------------------------------------------------------------- 5. Versionamiento
def test_segunda_version_conserva_la_primera_y_ambas_cargan(synthetic_csv, tmp_path):
    models = tmp_path / "models"
    r1 = entrenar(synthetic_csv, models, "v1")
    r2 = entrenar(synthetic_csv, models, "v2", n_estimators=20)
    assert r1["paths"]["model"].is_file() and r2["paths"]["model"].is_file()
    b1, b2 = load_version(models / "v1"), load_version(models / "v2")
    assert b1["version"] == "v1" and b2["version"] == "v2"
    assert (models / "v1" / "metrics.json").is_file() and (models / "v2" / "metadata.json").is_file()


def test_identificador_repetido_no_sobrescribe(synthetic_csv, tmp_path):
    models = tmp_path / "models"
    r1 = entrenar(synthetic_csv, models, "v1")
    antes = r1["paths"]["model"].read_bytes()
    meta_antes = r1["paths"]["metadata"].read_text(encoding="utf-8")
    with pytest.raises(VersionExistsError):
        entrenar(synthetic_csv, models, "v1", n_estimators=30)
    assert r1["paths"]["model"].read_bytes() == antes
    assert r1["paths"]["metadata"].read_text(encoding="utf-8") == meta_antes
    assert sorted(p.name for p in models.iterdir()) == ["v1"]  # sin carpetas temporales sobrantes


# ---------------------------------------------------------------- Validación del entrenamiento
def test_entrenamiento_sin_variable_objetivo_falla_con_mensaje_claro(synthetic_df, tmp_path):
    csv = tmp_path / "sin_objetivo.csv"
    synthetic_df.drop(columns=["unit_price"]).to_csv(csv, index=False)
    with pytest.raises(DataValidationError, match="unit_price"):
        entrenar(csv, tmp_path / "models", "v1")
