"""Carga de un modelo guardado y prediccion de un registro (Fase 3).

Reutiliza el esquema y el preprocesamiento de la Fase 2. Lo usan por igual
predict.py (CSV) y la ruta POST /predict de la API (un registro JSON).
"""
from __future__ import annotations
import os

import joblib
import pandas as pd

from .datos import validar_columnas, seleccionar_features, VALOR_POSITIVO


def cargar_modelo(ruta):
    """Acepta la ruta del .joblib o de la carpeta de version."""
    if os.path.isdir(ruta):
        ruta = os.path.join(ruta, "modelo.joblib")
    if not os.path.isfile(ruta):
        raise FileNotFoundError(f"No se encontro el modelo: {ruta}")
    return joblib.load(ruta)


def predecir_registro(modelo, registro):
    """Predice un unico registro (dict) y devuelve etiqueta y probabilidad.

    Construye una tabla de una fila en el orden del esquema, valida que esten
    las variables requeridas y aplica el mismo pipeline guardado.
    """
    df = pd.DataFrame([registro])
    validar_columnas(df, requiere_target=False)   # ValueError si falta una variable
    X = seleccionar_features(df)
    pred = int(modelo.predict(X)[0])
    proba = float(modelo.predict_proba(X)[0, 1])
    etiqueta = VALOR_POSITIVO if pred == 1 else "No"   # "Yes"/"No" como la variable Churn
    return {
        "prediccion": etiqueta,
        "clase": pred,
        "probabilidad_churn": round(proba, 6),
    }