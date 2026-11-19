"""Pruebas del registro de solicitudes y del script de monitoreo (Fase 4)."""
import contextlib
import csv
import io
from dataclasses import replace

import monitor
from api import create_app
from src.monitoring import compute_report, format_percentage, read_log
from src.request_log import HEADER, RequestLogger


def leer(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def correr_monitor(path):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = monitor.main(["--log", str(path)])
    return code, buf.getvalue()


# ------------------------------------------------------------------ registro
def test_cada_solicitud_genera_una_fila_incluida_la_rechazada(client_log, log_path, valid_record):
    assert client_log.post("/predict", json=valid_record).status_code == 200
    assert client_log.post("/predict", json=valid_record).status_code == 200
    incompleto = {k: v for k, v in valid_record.items() if k != "pizza_size"}
    assert client_log.post("/predict", json=incompleto).status_code == 422

    filas = leer(log_path)
    assert len(filas) == 3                                           # una sola fila por intento
    assert [f["estado_http"] for f in filas] == ["200", "200", "422"]
    assert [f["version"] for f in filas] == ["v1", "v1", ""]         # sin modelo usado -> versión vacía
    assert float(filas[0]["prediccion"]) > 0 and filas[2]["prediccion"] == ""
    assert filas[0]["fecha_hora"][:4].isdigit() and "T" in filas[0]["fecha_hora"]


def test_el_registro_conserva_el_historial_entre_ejecuciones(api_settings, log_path, valid_record):
    for _ in range(2):   # dos "ejecuciones" de la API sobre el mismo archivo
        app = create_app(replace(api_settings, log_path=log_path))
        app.test_client().post("/predict", json=valid_record)
    texto = log_path.read_text(encoding="utf-8").splitlines()
    assert texto[0] == ",".join(HEADER)
    assert texto.count(",".join(HEADER)) == 1                        # encabezado una sola vez
    assert len(texto) == 3                                           # encabezado + 2 filas


def test_se_registran_json_invalido_modelo_no_disponible_y_errores_de_tipo(api_settings, log_path, valid_record):
    app = create_app(replace(api_settings, log_path=log_path))
    c = app.test_client()
    c.post("/predict", data="no es json", content_type="application/json")      # 400
    c.post("/predict", json={**valid_record, "pizza_size": 7})                  # 422
    sin_modelo = create_app(replace(api_settings, model_version="v99", log_path=log_path)).test_client()
    sin_modelo.post("/predict", json=valid_record)                              # 503
    assert [f["estado_http"] for f in leer(log_path)] == ["400", "422", "503"]
    assert all(f["version"] == "" and f["prediccion"] == "" for f in leer(log_path))


def test_otros_endpoints_no_se_registran(client_log, log_path):
    client_log.get("/health")
    client_log.get("/models")
    client_log.get("/schema")
    assert not log_path.exists()


def test_un_fallo_del_registro_no_rompe_la_api(api_settings, tmp_path, valid_record):
    # log_path apunta a una carpeta: escribir falla, pero la predicción debe responderse igual
    app = create_app(replace(api_settings, log_path=tmp_path))
    assert app.test_client().post("/predict", json=valid_record).status_code == 200


def test_request_logger_escribe_prediccion_vacia_en_errores(tmp_path):
    logger = RequestLogger(tmp_path / "x.csv")
    logger.log(422)
    logger.log(200, "v1", 18.7)
    filas = leer(tmp_path / "x.csv")
    assert filas[0]["version"] == "" and filas[0]["prediccion"] == ""
    assert filas[1]["version"] == "v1" and filas[1]["prediccion"] == "18.7"


# ------------------------------------------------------------------ script de monitoreo
def test_monitor_dos_validas_y_una_invalida_da_3_solicitudes_y_33_3_por_ciento(client_log, log_path, valid_record):
    client_log.post("/predict", json=valid_record)
    client_log.post("/predict", json=valid_record)
    client_log.post("/predict", json={"pizza_size": "M"})
    code, salida = correr_monitor(log_path)
    assert code == 0
    assert "Total de solicitudes: 3" in salida
    assert "Porcentaje de solicitudes fallidas: 33,3 %" in salida
    assert "v1: 2" in salida


def test_monitor_con_registro_vacio_muestra_total_0_y_sin_solicitudes(tmp_path):
    vacio = tmp_path / "vacio.csv"
    vacio.write_text(",".join(HEADER) + "\n", encoding="utf-8")          # solo encabezado
    code, salida = correr_monitor(vacio)
    assert code == 0 and "Total de solicitudes: 0" in salida and "sin solicitudes" in salida
    completamente_vacio = tmp_path / "cero.csv"
    completamente_vacio.write_text("", encoding="utf-8")
    code, salida = correr_monitor(completamente_vacio)
    assert code == 0 and "Total de solicitudes: 0" in salida and "sin solicitudes" in salida


def test_compute_report_cuenta_4xx_y_5xx_como_fallidas():
    filas = [{"estado_http": c, "version": v} for c, v in
             [("200", "v1"), ("201", "v1"), ("400", ""), ("422", ""), ("503", ""), ("500", "")]]
    r = compute_report(filas)
    assert r["total"] == 6 and r["fallidas"] == 4
    assert format_percentage(r["porcentaje"]) == "66,7 %"
    assert format_percentage(None) == "sin solicitudes"


def test_monitor_con_archivo_inexistente_falla_con_codigo_distinto_de_cero(tmp_path):
    code, _ = correr_monitor(tmp_path / "no_existe.csv")
    assert code != 0


def test_read_log_rechaza_un_archivo_sin_las_columnas_esperadas(tmp_path):
    from src.errors import AppError
    import pytest
    malo = tmp_path / "malo.csv"
    malo.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(AppError, match="columnas"):
        read_log(malo)


def test_la_version_activa_por_defecto_en_fase_4_es_v2_y_se_puede_volver_a_v1(tmp_path):
    from src.settings import Settings
    assert Settings.from_env(tmp_path, {}).model_version == "v2"
    assert Settings.from_env(tmp_path, {"MODEL_VERSION": "v1"}).model_version == "v1"
