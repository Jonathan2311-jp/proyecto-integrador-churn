"""Construccion del Pipeline completo y calculo de metricas."""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from .datos import limpiar_determinista
from .preprocesamiento import construir_preprocesador

ALGORITMO = "LogisticRegression"


def construir_modelo(C=0.1, seed=42):
    """Pipeline autosuficiente: limpieza -> preprocesamiento -> regresion logistica.

    La limpieza determinista viaja dentro del artefacto, asi al cargarlo con
    joblib no hace falta repetir ninguna transformacion externa.
    """
    return Pipeline([
        ("limpieza", FunctionTransformer(limpiar_determinista)),
        ("prep", construir_preprocesador(escalar=True)),
        ("clf", LogisticRegression(C=C, max_iter=1000,
                                   class_weight="balanced", random_state=seed)),
    ])


def calcular_metricas(y_true, y_pred, y_proba):
    """Metricas de clasificacion sobre la clase positiva (churn)."""
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_true, y_proba)),
    }