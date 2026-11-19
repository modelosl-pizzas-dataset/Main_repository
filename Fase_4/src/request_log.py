"""Registro persistente de las solicitudes a POST /predict (una fila por solicitud, CSV en modo agregar)."""
import csv
import threading
from datetime import datetime
from pathlib import Path

HEADER = ["fecha_hora", "estado_http", "version", "prediccion"]


class RequestLogger:
    """Escribe un evento por solicitud. El encabezado se escribe solo al crear el archivo;
    el historial se conserva entre ejecuciones (apertura en modo 'a')."""

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def log(self, estado_http, version=None, prediccion=None, when=None):
        """Agrega una fila. En un error, `version` y `prediccion` quedan vacías."""
        when = when or datetime.now()
        row = [when.isoformat(timespec="seconds"), int(estado_http),
               "" if version is None else version, "" if prediccion is None else prediccion]
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            needs_header = (not self.path.exists()) or self.path.stat().st_size == 0
            with open(self.path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh, lineterminator="\n")
                if needs_header:
                    writer.writerow(HEADER)
                writer.writerow(row)
