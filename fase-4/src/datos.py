"""Lectura, validacion y limpieza determinista de los datos de churn.

Centraliza TODO lo que depende del esquema del problema, para que `train.py`
y `predict.py` usen exactamente las mismas reglas (sin duplicar codigo).
"""
from __future__ import annotations
import pandas as pd

# --- Esquema del problema (heredado de la Fase 1) ---
COLUMNA_ID = "customerID"
TARGET = "Churn"
VALOR_POSITIVO = "Yes"            # se codifica como 1

COLS_NUM = ["tenure", "MonthlyCharges", "TotalCharges"]
COLS_BIN = ["SeniorCitizen"]      # ya viene codificada 0/1
COLS_CAT = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod"]

# Orden canonico de las 19 variables de entrada
FEATURES = COLS_NUM + COLS_BIN + COLS_CAT


def cargar_csv(ruta):
    """Lee un CSV y devuelve un DataFrame. Error claro si no se puede leer."""
    try:
        return pd.read_csv(ruta)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"No se encontro el archivo de datos: {ruta}") from e


def validar_columnas(df, requiere_target=False):
    """Verifica que esten las 19 variables de entrada (y el target si se pide).

    Lanza ValueError con un mensaje claro si falta alguna. Es la defensa que
    permite que predict.py termine con codigo distinto de cero ante una entrada
    invalida (seccion 7.3 de la especificacion).
    """
    faltantes = [c for c in FEATURES if c not in df.columns]
    if faltantes:
        raise ValueError("Faltan columnas requeridas: " + ", ".join(faltantes))
    if requiere_target and TARGET not in df.columns:
        raise ValueError(f"Falta la columna objetivo '{TARGET}' en los datos.")


def validar_tipos(df):
    #Valida tipos y nulos de las variables de entrada antes de predecir. Indicando la columna y fila del problema

    problemas = []
    for c in COLS_NUM:
        serie = pd.to_numeric(df[c], errors="coerce")
        vacia = df[c].isna() | (df[c].astype(str).str.strip() == "")
        no_numerica = serie.isna() & ~vacia
        if no_numerica.any():
            filas = (df.index[no_numerica] + 2).tolist()[:5]
            problemas.append(f"'{c}' debe ser numerica; hay valores no numericos "
                             f"en las filas {filas} del CSV")
        elif c != "TotalCharges" and vacia.any():
            problemas.append(f"'{c}' tiene {int(vacia.sum())} valores nulos y no admite "
                             "nulos (solo TotalCharges se imputa con 0)")
    if not pd.to_numeric(df["SeniorCitizen"], errors="coerce").isin([0, 1]).all():
        problemas.append("'SeniorCitizen' solo admite los valores 0 o 1")
    if problemas:
        raise ValueError("Entrada invalida:\n  - " + "\n  - ".join(problemas))


def limpiar_determinista(df):
    """Transformaciones que NO aprenden de los datos (no generan fuga).

    Viaja DENTRO del Pipeline, de modo que el artefacto guardado es
    autosuficiente (correccion respecto de la Fase 1, donde este paso se hacia
    fuera, en el notebook).

    - TotalCharges: texto -> numerico. Los clientes nuevos (tenure=0) aun no
      tienen factura acumulada, quedan como NaN y se rellenan con 0.
    """
    X = df.copy()
    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0)
    return X


def seleccionar_features(df):
    """Devuelve solo las 19 variables de entrada, en el orden canonico."""
    return df[FEATURES].copy()


def separar_X_y(df):
    """Codifica el objetivo (Yes->1, No->0) y separa predictoras y objetivo."""
    y = (df[TARGET] == VALOR_POSITIVO).astype(int)
    X = seleccionar_features(df)
    return X, y