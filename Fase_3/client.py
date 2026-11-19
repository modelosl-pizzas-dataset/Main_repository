"""Cliente de prueba sencillo para la API (solo usa la librería estándar).

Ejemplos (con la API ya iniciada en el puerto 8000):
    python client.py                        # /health, /models y una predicción por cada fila de data/ejemplo_prediccion.csv
    python client.py --incluir-invalida     # además envía una solicitud sin un campo obligatorio (debe dar 422)
    python client.py --train                # además llama a POST /train (tarda unos segundos)
"""
import argparse
import csv
import json
import sys
import urllib.error
import urllib.request


def call(base_url, method, path, payload=None):
    """Envía una solicitud y devuelve (código_http, cuerpo_json)."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(base_url.rstrip("/") + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # 4xx / 5xx: la API respondió con un JSON de error
        return exc.code, json.loads(exc.read().decode("utf-8"))


def rows_from_csv(path):
    """Convierte cada fila del CSV de ejemplo en un registro JSON (celda vacía -> null)."""
    with open(path, newline="", encoding="utf-8") as fh:
        return [{k: (v if v != "" else None) for k, v in row.items()} for row in csv.DictReader(fh)]


def show(title, method, path, payload, status, body):
    print(f"\n### {title}\n{method} {path}")
    if payload is not None:
        print("Solicitud :", json.dumps(payload, ensure_ascii=False))
    print(f"Respuesta : HTTP {status}", json.dumps(body, ensure_ascii=False))


def main(argv=None):
    p = argparse.ArgumentParser(description="Cliente de prueba de la API de predicción de precio de pizzas.")
    p.add_argument("--url", default="http://localhost:8000", help="URL base de la API (default http://localhost:8000)")
    p.add_argument("--input", default="data/ejemplo_prediccion.csv", help="CSV con registros de ejemplo")
    p.add_argument("--filas", type=int, default=None, help="Enviar solo las primeras N filas del CSV")
    p.add_argument("--incluir-invalida", action="store_true", help="Enviar además un registro al que le falta un campo")
    p.add_argument("--train", action="store_true", help="Llamar también a POST /train (cuerpo vacío)")
    args = p.parse_args(argv)

    try:
        status, body = call(args.url, "GET", "/health")
        show("Estado de la API", "GET", "/health", None, status, body)
        status, body = call(args.url, "GET", "/models")
        show("Versiones disponibles", "GET", "/models", None, status, body)

        registros = rows_from_csv(args.input)[: args.filas]
        for i, registro in enumerate(registros, 1):
            status, body = call(args.url, "POST", "/predict", registro)
            show(f"Predicción {i}", "POST", "/predict", registro, status, body)

        if args.incluir_invalida:
            incompleto = {k: v for k, v in registros[0].items() if k != "pizza_ingredients"}
            status, body = call(args.url, "POST", "/predict", incompleto)
            show("Registro inválido (falta 'pizza_ingredients')", "POST", "/predict", incompleto, status, body)

        if args.train:
            status, body = call(args.url, "POST", "/train")
            show("Entrenamiento de una nueva versión", "POST", "/train", None, status, body)
    except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
        print(f"ERROR: no se pudo conectar con {args.url}: {exc}\n"
              "¿La API está iniciada? (python -m flask --app api run --port 8000)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
