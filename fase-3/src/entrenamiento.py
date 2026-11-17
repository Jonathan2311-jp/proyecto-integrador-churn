"""Logica de entrenamiento reutilizable (Fase 3).

Extrae el nucleo de train.py a una funcion, para que la usen por igual el
script de linea de comandos (train.py) y la ruta POST /train de la API.
"""
from __future__ import annotations
import json
import os
import platform
import subprocess
from datetime import datetime, timezone

import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.dummy import DummyClassifier

from .datos import (cargar_csv, validar_columnas, separar_X_y,
                    FEATURES, COLS_NUM, COLS_BIN, COLS_CAT, TARGET)
from .modelo import construir_modelo, calcular_metricas, ALGORITMO


def _hash_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "desconocido"


def _tipos_features():
    tipos = {c: "numerica" for c in COLS_NUM}
    tipos.update({c: "binaria (0/1)" for c in COLS_BIN})
    tipos.update({c: "categorica" for c in COLS_CAT})
    return tipos


def entrenar_y_guardar(data, target=TARGET, output_dir="models", version="v1",
                       seed=42, test_size=0.20, cv_folds=5):
    """Entrena, evalua contra el modelo base y guarda una version.

    Devuelve un resumen (dict) con la version, las rutas y las metricas.
    Lanza FileExistsError si la version ya existe (no sobrescribe).
    """
    dir_version = os.path.join(output_dir, version)
    if os.path.isdir(dir_version) and os.listdir(dir_version):
        raise FileExistsError(
            f"La version '{version}' ya existe en {dir_version}. "
            "Use otro identificador para no sobrescribirla.")

    df = cargar_csv(data)
    if target != TARGET:
        if target not in df.columns:
            raise ValueError(f"No existe la columna objetivo '{target}'.")
        df = df.rename(columns={target: TARGET})
    validar_columnas(df, requiere_target=True)
    X, y = separar_X_y(df)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    busqueda = GridSearchCV(construir_modelo(seed=seed),
                            {"clf__C": [0.01, 0.1, 1, 10]},
                            scoring="f1", cv=cv, n_jobs=-1, refit=True)
    busqueda.fit(Xtr, ytr)
    modelo = busqueda.best_estimator_
    mejor_C = busqueda.best_params_["clf__C"]

    yp = modelo.predict(Xte)
    ypr = modelo.predict_proba(Xte)[:, 1]
    met_modelo = calcular_metricas(yte, yp, ypr)

    base = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    met_base = calcular_metricas(yte, base.predict(Xte), base.predict_proba(Xte)[:, 1])

    os.makedirs(dir_version, exist_ok=True)
    ruta_modelo = os.path.join(dir_version, "modelo.joblib")
    joblib.dump(modelo, ruta_modelo)

    metrics = {
        "metrica_principal": "f1",
        "modelo": met_modelo,
        "modelo_base": met_base,
        "f1_validacion_cruzada": float(busqueda.best_score_),
    }
    with open(os.path.join(dir_version, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    metadata = {
        "version": version,
        "fecha_creacion": datetime.now(timezone.utc).isoformat(),
        "tipo_problema": "clasificacion binaria",
        "variable_objetivo": TARGET,
        "algoritmo": ALGORITMO,
        "hiperparametros": {"C": mejor_C, "class_weight": "balanced", "max_iter": 1000},
        "semilla": seed,
        "variables_entrada": FEATURES,
        "tipos_variables": _tipos_features(),
        "datos": {"archivo": os.path.basename(data), "n_observaciones": int(len(df))},
        "particion": {
            "estrategia": "estratificada por el objetivo",
            "test_size": test_size,
            "n_entrenamiento": int(len(Xtr)),
            "n_prueba": int(len(Xte)),
            "validacion_cruzada": f"StratifiedKFold({cv_folds})",
        },
        "metrica_evaluacion": "f1 (clase positiva) y ROC-AUC",
        "python": platform.python_version(),
        "dependencias": "requirements.txt",
        "commit": _hash_commit(),
    }
    with open(os.path.join(dir_version, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return {
        "version": version,
        "dir": dir_version,
        "mejor_C": mejor_C,
        "f1": met_modelo["f1"],
        "roc_auc": met_modelo["roc_auc"],
        "f1_validacion_cruzada": float(busqueda.best_score_),
    }