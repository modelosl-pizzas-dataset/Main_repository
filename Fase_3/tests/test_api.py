"""Pruebas de la API (Fase 3). Usan el cliente de pruebas de Flask y un modelo pequeño en carpetas temporales."""
import pytest

from api import create_app
from src.settings import Settings


# ------------------------------------------------------------------ las tres pruebas exigidas
def test_health_responde_200_y_estado(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok"
    assert body["modelo_activo"] == "v1" and body["modelo_disponible"] is True


def test_predict_entrada_valida_devuelve_prediccion_y_version(client, valid_record):
    r = client.post("/predict", json=valid_record)
    assert r.status_code == 200
    body = r.get_json()
    assert set(body) == {"prediccion", "version"}
    assert body["version"] == "v1"
    assert isinstance(body["prediccion"], float) and body["prediccion"] > 0


def test_predict_con_campo_obligatorio_ausente_se_rechaza(client, valid_record):
    del valid_record["pizza_category"]
    r = client.post("/predict", json=valid_record)
    assert r.status_code in (400, 422)
    body = r.get_json()
    assert "pizza_category" in body["error"]          # el mensaje identifica el campo
    assert body["campos"] == ["pizza_category"]
    assert "prediccion" not in body


# ------------------------------------------------------------------ validación adicional
def test_predict_tipo_incompatible_se_rechaza(client, valid_record):
    valid_record["pizza_size"] = 12
    r = client.post("/predict", json=valid_record)
    assert r.status_code == 422
    assert "pizza_size" in r.get_json()["error"]


def test_predict_tamano_no_permitido_se_rechaza(client, valid_record):
    valid_record["pizza_size"] = "XS"
    r = client.post("/predict", json=valid_record)
    assert r.status_code == 422 and "XS" in r.get_json()["error"]


def test_predict_acepta_nulos_y_los_imputa(client):
    r = client.post("/predict", json={"pizza_size": None, "pizza_category": None, "pizza_ingredients": None})
    assert r.status_code == 200
    assert r.get_json()["prediccion"] > 0


def test_predict_ignora_campos_extra_y_los_informa(client, valid_record):
    valid_record["unit_price"] = 99          # la variable objetivo no debe influir
    sin_extra = client.post("/predict", json={k: v for k, v in valid_record.items() if k != "unit_price"}).get_json()
    con_extra = client.post("/predict", json=valid_record).get_json()
    assert con_extra["campos_ignorados"] == ["unit_price"]
    assert con_extra["prediccion"] == sin_extra["prediccion"]


def test_predict_con_json_invalido_devuelve_400(client):
    r = client.post("/predict", data="esto no es json", content_type="application/json")
    assert r.status_code == 400


# ------------------------------------------------------------------ modelo no disponible
def test_predict_sin_modelo_disponible_devuelve_503(api_settings, valid_record):
    from dataclasses import replace
    app = create_app(replace(api_settings, model_version="v99"))
    r = app.test_client().post("/predict", json=valid_record)
    assert r.status_code == 503
    msg = r.get_json()["error"]
    assert "v99" in msg and "v1" in msg and "MODEL_VERSION" in msg      # indica cómo elegir una versión disponible
    assert app.test_client().get("/health").get_json()["modelo_disponible"] is False


def test_seleccionar_otra_version_cambia_la_version_usada(api_settings, client, valid_record):
    from dataclasses import replace
    # entrenar v2 vía API y luego configurarla como activa
    assert client.post("/train", json={"n_estimators": 5}).status_code == 201
    app2 = create_app(replace(api_settings, model_version="v2"))
    assert app2.test_client().post("/predict", json=valid_record).get_json()["version"] == "v2"


# ------------------------------------------------------------------ /train
def test_train_sin_cuerpo_crea_nueva_version_y_conserva_las_anteriores(client, api_settings):
    antes = (api_settings.models_dir / "v1" / "modelo.joblib").read_bytes()
    r = client.post("/train")                      # cuerpo vacío
    assert r.status_code == 201
    body = r.get_json()
    assert body["version"] == "v2"
    for nombre in ("modelo.joblib", "metrics.json", "metadata.json"):
        assert (api_settings.models_dir / "v2" / nombre).is_file()
    assert (api_settings.models_dir / "v1" / "modelo.joblib").read_bytes() == antes   # v1 intacta
    assert body["version_activa"] == "v1"
    assert {"modelo", "modelo_base"} <= set(body["metricas"])


def test_train_con_version_repetida_devuelve_409_sin_sobrescribir(client, api_settings):
    antes = (api_settings.models_dir / "v1" / "modelo.joblib").read_bytes()
    r = client.post("/train", json={"version": "v1"})
    assert r.status_code == 409
    assert (api_settings.models_dir / "v1" / "modelo.joblib").read_bytes() == antes


def test_train_con_parametros_invalidos_devuelve_422(client):
    assert client.post("/train", json={"n_estimators": "muchos"}).status_code == 422
    assert client.post("/train", json={"parametro_raro": 1}).status_code == 422
    assert client.post("/train", json={"version": "../hack"}).status_code == 422


def test_train_sin_datos_devuelve_error_claro(api_settings):
    from dataclasses import replace
    app = create_app(replace(api_settings, data_path=api_settings.data_path.parent / "no_existe.csv"))
    r = app.test_client().post("/train")
    assert r.status_code == 500 and "no_existe.csv" in r.get_json()["error"]


# ------------------------------------------------------------------ otros endpoints (listos para un frontend)
def test_models_lista_versiones_y_marca_la_activa(client):
    body = client.get("/models").get_json()
    assert body["version_activa"] == "v1"
    assert [v["version"] for v in body["versiones"]] == ["v1"]
    assert body["versiones"][0]["activa"] is True and body["versiones"][0]["rmse"] is not None


def test_schema_describe_las_variables_de_entrada(client):
    body = client.get("/schema").get_json()
    nombres = [v["nombre"] for v in body["variables"]]
    assert nombres == ["pizza_size", "pizza_category", "pizza_ingredients"]
    assert body["variables"][0]["valores_permitidos"] == ["S", "M", "L", "XL", "XXL"]


def test_cors_permite_consumo_desde_un_frontend(client):
    assert client.get("/health").headers["Access-Control-Allow-Origin"] == "*"
    assert client.options("/predict").status_code == 200


def test_rutas_inexistentes_responden_json(client):
    r = client.get("/no-existe")
    assert r.status_code == 404 and "error" in r.get_json()


def test_settings_desde_entorno(tmp_path):
    s = Settings.from_env(tmp_path, {"MODEL_VERSION": "v2", "TRAIN_SEED": "7", "MODELS_DIR": "m"})
    assert s.model_version == "v2" and s.seed == 7 and s.models_dir == tmp_path / "m"
    with pytest.raises(ValueError):
        Settings.from_env(tmp_path, {"MODEL_VERSION": "../x"})
