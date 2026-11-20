"""Cliente de prueba de la API de churn (Fase 3).

Lee un registro de data/ejemplo_prediccion.csv, lo convierte a JSON y lo envia
a POST /predict. Ejemplos:

    python cliente.py                          # primera fila, API en localhost:8000
    python cliente.py --fila 2 --url http://localhost:8000
"""
import argparse
import csv
import json
import sys
import urllib.error
import urllib.request

ENTEROS = {"tenure", "SeniorCitizen"}
DECIMALES = {"MonthlyCharges", "TotalCharges"}


def leer_registro(ruta, fila):
    """Devuelve la fila indicada del CSV como dict con los tipos que espera la API."""
    with open(ruta, newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    if not 0 <= fila < len(filas):
        raise SystemExit(f"ERROR: la fila {fila} no existe (el archivo tiene {len(filas)}).")
    registro = {}
    for campo, valor in filas[fila].items():
        if campo == "customerID":
            continue                      # identificador: no es variable predictora
        if campo in ENTEROS:
            registro[campo] = int(valor)
        elif campo in DECIMALES:
            registro[campo] = float(valor) if valor.strip() else None  # vacio -> null
        else:
            registro[campo] = valor
    return registro


def enviar(url, registro):
    """POST /predict. Devuelve (codigo HTTP, cuerpo JSON)."""
    peticion = urllib.request.Request(
        url.rstrip("/") + "/predict",
        data=json.dumps(registro).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(peticion, timeout=30) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)
    except urllib.error.URLError as e:
        raise SystemExit(f"ERROR: no se pudo conectar con la API en {url}: {e.reason}")


def main():
    p = argparse.ArgumentParser(description="Cliente de prueba de la API de churn")
    p.add_argument("--url", default="http://localhost:8000", help="direccion de la API")
    p.add_argument("--archivo", default="data/ejemplo_prediccion.csv")
    p.add_argument("--fila", type=int, default=0, help="fila del CSV (desde 0)")
    args = p.parse_args()

    registro = leer_registro(args.archivo, args.fila)
    print("Solicitud  POST /predict:")
    print(json.dumps(registro, indent=2))
    codigo, cuerpo = enviar(args.url, registro)
    print(f"\nRespuesta  HTTP {codigo}:")
    print(json.dumps(cuerpo, indent=2, ensure_ascii=False))
    sys.exit(0 if codigo == 200 else 1)


if __name__ == "__main__":
    main()