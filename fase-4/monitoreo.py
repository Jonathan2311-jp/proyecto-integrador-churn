"""Metricas de operacion de la API a partir del registro de solicitudes (Fase 4).

Cuenta las solicitudes y el porcentaje de fallidas (respuestas HTTP 4xx o 5xx).
Describe el funcionamiento del servicio, no la calidad predictiva del modelo.

    python monitoreo.py
    python monitoreo.py --registro logs/solicitudes.csv
"""
import argparse
import csv
import os
import sys


def leer_estados(ruta):
    """Devuelve la lista de codigos HTTP del registro."""
    estados = []
    with open(ruta, newline="", encoding="utf-8") as f:
        for n, fila in enumerate(csv.DictReader(f), start=2):
            try:
                estados.append(int(fila["estado_http"]))
            except (KeyError, TypeError, ValueError):
                raise SystemExit(f"ERROR: estado_http invalido en la linea {n} de {ruta}.")
    return estados


def calcular(estados):
    """Devuelve (total, fallidas, porcentaje); el porcentaje es None si no hay solicitudes."""
    total = len(estados)
    fallidas = sum(1 for e in estados if 400 <= e <= 599)
    return total, fallidas, (fallidas / total * 100 if total else None)


def main():
    p = argparse.ArgumentParser(description="Metricas de operacion de la API")
    p.add_argument("--registro", default="logs/solicitudes.csv",
                   help="archivo CSV de solicitudes (por defecto logs/solicitudes.csv)")
    args = p.parse_args()

    if os.path.exists(args.registro):
        estados = leer_estados(args.registro)
    else:
        print(f"AVISO: no existe {args.registro}; se considera un registro vacio.",
              file=sys.stderr)
        estados = []
    total, fallidas, porcentaje = calcular(estados)
    print(f"Total de solicitudes: {total}")
    print(f"Solicitudes fallidas (4xx/5xx): {fallidas}")
    if porcentaje is None:
        print("Porcentaje de fallos: sin solicitudes")
    else:
        print(f"Porcentaje de fallos: {porcentaje:.1f} %".replace(".", ","))


if __name__ == "__main__":
    main()