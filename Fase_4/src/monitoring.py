"""Métricas de operación calculadas a partir del registro de solicitudes."""
import csv
from collections import Counter
from pathlib import Path

from src.errors import AppError
from src.request_log import HEADER


def read_log(path):
    path = Path(path)
    if not path.is_file():
        raise AppError(f"No existe el registro de solicitudes: {path}")
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:          # archivo completamente vacío
            return []
        faltan = [c for c in HEADER if c not in reader.fieldnames]
        if faltan:
            raise AppError(f"El registro '{path}' no tiene las columnas esperadas; faltan: {faltan}")
        return list(reader)


def compute_report(rows):
    """total de solicitudes, fallidas (HTTP 4xx o 5xx) y porcentaje (None si no hay solicitudes)."""
    total = len(rows)
    fallidas = 0
    ilegibles = 0
    por_version = Counter()
    for row in rows:
        try:
            code = int(row["estado_http"])
        except (TypeError, ValueError):
            ilegibles += 1
            code = None
        if code is not None and 400 <= code <= 599:
            fallidas += 1
        por_version[row.get("version") or "(sin modelo)"] += 1
    porcentaje = None if total == 0 else fallidas / total * 100
    return {"total": total, "fallidas": fallidas, "porcentaje": porcentaje,
            "ilegibles": ilegibles, "por_version": dict(por_version)}


def format_percentage(value):
    """33.333… -> '33,3 %'; None -> 'sin solicitudes'."""
    if value is None:
        return "sin solicitudes"
    return f"{value:.1f}".replace(".", ",") + " %"
