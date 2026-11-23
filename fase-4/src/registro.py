"""Registro de solicitudes a /predict en un archivo CSV (Fase 4)."""
import csv
import os
import threading
from datetime import datetime, timezone

CAMPOS = ["fecha_hora", "estado_http", "version", "prediccion"]
_bloqueo = threading.Lock()


def registrar_solicitud(ruta, estado_http, version="", prediccion=""):
    """Agrega una fila al registro (hora UTC).

    Crea la carpeta y el archivo con su encabezado si no existen, y abre el
    archivo en modo de agregar para conservar el historial entre ejecuciones.
    """
    carpeta = os.path.dirname(ruta)
    if carpeta:
        os.makedirs(carpeta, exist_ok=True)
    fila = [datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            estado_http, version, prediccion]
    with _bloqueo:
        nuevo = not os.path.exists(ruta) or os.path.getsize(ruta) == 0
        with open(ruta, "a", newline="", encoding="utf-8") as f:
            escritor = csv.writer(f, lineterminator="\n")
            if nuevo:
                escritor.writerow(CAMPOS)
            escritor.writerow(fila)