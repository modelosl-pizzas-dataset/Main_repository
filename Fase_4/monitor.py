"""Script de monitoreo: lee el registro de /predict y muestra dos métricas de operación.

    python monitor.py --log logs/predict_log.csv

- Total de solicitudes (filas de datos del registro).
- Porcentaje de solicitudes fallidas (HTTP 4xx o 5xx). Con el registro vacío: total 0 y «sin solicitudes».
"""
import argparse
import sys

from src.errors import AppError
from src.monitoring import compute_report, format_percentage, read_log


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Muestra el total de solicitudes y el porcentaje de fallos de la API.")
    p.add_argument("--log", default="logs/predict_log.csv", help="CSV de registro (default: logs/predict_log.csv)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        report = compute_report(read_log(args.log))
    except AppError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Registro: {args.log}")
    print(f"Total de solicitudes: {report['total']}")
    print(f"Solicitudes fallidas (HTTP 4xx/5xx): {report['fallidas']}")
    print(f"Porcentaje de solicitudes fallidas: {format_percentage(report['porcentaje'])}")
    if report["por_version"]:
        detalle = ", ".join(f"{v}: {n}" for v, n in sorted(report["por_version"].items()))
        print(f"Solicitudes por versión: {detalle}")
    if report["ilegibles"]:
        print(f"Aviso: {report['ilegibles']} fila(s) con código HTTP ilegible (no cuentan como fallidas).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
