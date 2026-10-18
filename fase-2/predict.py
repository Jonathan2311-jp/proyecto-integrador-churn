"""Prediccion con una version guardada del modelo de churn (Fase 2).

Carga un modelo ya entrenado y genera predicciones sobre un CSV de entrada,
sin reentrenar ni reajustar ninguna transformacion.

Uso:
    python predict.py --model models/v1/modelo.joblib \
        --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
"""
from __future__ import annotations
import argparse
import os
import sys

import joblib

from src.datos import cargar_csv, validar_columnas, seleccionar_features


def resolver_modelo(ruta):
    """Acepta la ruta del .joblib o de la carpeta de version (usa modelo.joblib)."""
    if os.path.isdir(ruta):
        ruta = os.path.join(ruta, "modelo.joblib")
    if not os.path.isfile(ruta):
        raise FileNotFoundError(f"No se encontro el modelo: {ruta}")
    return ruta


def parse_args():
    p = argparse.ArgumentParser(description="Predice churn con una version guardada del modelo.")
    p.add_argument("--model", required=True,
                   help="Ruta del modelo .joblib o de la carpeta de version (ej: models/v1).")
    p.add_argument("--input", required=True,
                   help="CSV con las variables de entrada (no requiere la columna objetivo).")
    p.add_argument("--output", required=True, help="Ruta del CSV de salida.")
    return p.parse_args()


def main():
    args = parse_args()

    ruta_modelo = resolver_modelo(args.model)
    modelo = joblib.load(ruta_modelo)

    df = cargar_csv(args.input)
    # Valida que esten las 19 variables. Una celda nula NO es una columna ausente.
    validar_columnas(df, requiere_target=False)

    # Ordena al esquema del entrenamiento e ignora columnas adicionales.
    # Categorias no vistas se ignoran (OneHotEncoder handle_unknown='ignore').
    X = seleccionar_features(df)
    pred = modelo.predict(X)
    proba = modelo.predict_proba(X)[:, 1]

    salida = df.copy()                 # conserva columnas y orden original (trazabilidad)
    salida["prediccion"] = pred        # 1 = Cancela (Churn=Yes), 0 = No cancela
    salida["prediccion_etiqueta"] = ["Cancela" if p == 1 else "No cancela" for p in pred]
    salida["probabilidad_churn"] = proba.round(6)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    salida.to_csv(args.output, index=False)

    print("=== Prediccion completada ===")
    print(f"Modelo:  {ruta_modelo}")
    print(f"Entrada: {args.input}  ({len(df)} filas)")
    print(f"Salida:  {args.output}")
    print(f"Marcados como 'Cancela' (1): {int(pred.sum())} de {len(pred)}")
    print("Etiquetas: 1 = el cliente cancelaria (churn), 0 = permaneceria.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)