"""Entrenamiento de la aplicacion de churn (Fase 2).

Entrena el modelo, lo evalua contra un modelo base y guarda una version
reutilizable (modelo.joblib + metrics.json + metadata.json).

Uso:
    python train.py --data data/telco_churn.csv --target Churn \
        --output-dir models --version v1 --seed 42
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.dummy import DummyClassifier

from src.datos import (cargar_csv, validar_columnas, separar_X_y,
                       FEATURES, COLS_NUM, COLS_BIN, COLS_CAT, TARGET)
from src.modelo import construir_modelo, calcular_metricas, ALGORITMO


def hash_commit():
    """Hash del commit actual, para identificar el codigo usado (seccion 10.1)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "desconocido"


def tipos_features():
    tipos = {c: "numerica" for c in COLS_NUM}
    tipos.update({c: "binaria (0/1)" for c in COLS_BIN})
    tipos.update({c: "categorica" for c in COLS_CAT})
    return tipos


def parse_args():
    p = argparse.ArgumentParser(description="Entrena y versiona el modelo de churn.")
    p.add_argument("--data", required=True, help="Ruta del CSV etiquetado.")
    p.add_argument("--target", default=TARGET, help=f"Variable objetivo (def: {TARGET}).")
    p.add_argument("--output-dir", default="models", help="Carpeta base de versiones (def: models).")
    p.add_argument("--version", required=True, help="Identificador de version, ej: v1.")
    p.add_argument("--seed", type=int, default=42, help="Semilla aleatoria (def: 42).")
    p.add_argument("--test-size", type=float, default=0.20, help="Proporcion de prueba (def: 0.20).")
    p.add_argument("--cv-folds", type=int, default=5, help="Particiones de CV (def: 5).")
    return p.parse_args()


def main():
    args = parse_args()

    # No sobrescribir una version existente (seccion 10.2)
    dir_version = os.path.join(args.output_dir, args.version)
    if os.path.isdir(dir_version) and os.listdir(dir_version):
        raise FileExistsError(
            f"La version '{args.version}' ya existe en {dir_version}. "
            "Use otro identificador para no sobrescribirla.")

    # 1) Datos
    df = cargar_csv(args.data)
    if args.target != TARGET:
        if args.target not in df.columns:
            raise ValueError(f"No existe la columna objetivo '{args.target}'.")
        df = df.rename(columns={args.target: TARGET})
    validar_columnas(df, requiere_target=True)
    X, y = separar_X_y(df)

    # 2) Particion estratificada ANTES de ajustar transformaciones
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed)

    # 3) Hiperparametros por CV, solo sobre entrenamiento
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    busqueda = GridSearchCV(construir_modelo(seed=args.seed),
                            {"clf__C": [0.01, 0.1, 1, 10]},
                            scoring="f1", cv=cv, n_jobs=-1, refit=True)
    busqueda.fit(Xtr, ytr)
    modelo = busqueda.best_estimator_
    mejor_C = busqueda.best_params_["clf__C"]

    # 4) Evaluacion en test
    yp = modelo.predict(Xte)
    ypr = modelo.predict_proba(Xte)[:, 1]
    met_modelo = calcular_metricas(yte, yp, ypr)

    # 5) Modelo base con la misma particion y metrica
    base = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    met_base = calcular_metricas(yte, base.predict(Xte), base.predict_proba(Xte)[:, 1])

    # 6) Guardado de artefactos
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
        "version": args.version,
        "fecha_creacion": datetime.now(timezone.utc).isoformat(),
        "tipo_problema": "clasificacion binaria",
        "variable_objetivo": TARGET,
        "algoritmo": ALGORITMO,
        "hiperparametros": {"C": mejor_C, "class_weight": "balanced", "max_iter": 1000},
        "semilla": args.seed,
        "variables_entrada": FEATURES,
        "tipos_variables": tipos_features(),
        "datos": {"archivo": os.path.basename(args.data), "n_observaciones": int(len(df))},
        "particion": {
            "estrategia": "estratificada por el objetivo",
            "test_size": args.test_size,
            "n_entrenamiento": int(len(Xtr)),
            "n_prueba": int(len(Xte)),
            "validacion_cruzada": f"StratifiedKFold({args.cv_folds})",
        },
        "metrica_evaluacion": "f1 (clase positiva) y ROC-AUC",
        "python": platform.python_version(),
        "dependencias": "requirements.txt",
        "commit": hash_commit(),
    }
    with open(os.path.join(dir_version, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # 7) Resumen
    print("=== Entrenamiento completado ===")
    print(f"Version:      {args.version}")
    print(f"Mejor C (CV): {mejor_C}  | F1 CV: {busqueda.best_score_:.4f}")
    print(f"Modelo -> F1: {met_modelo['f1']:.4f} | ROC-AUC: {met_modelo['roc_auc']:.4f} | "
          f"recall: {met_modelo['recall']:.4f}")
    print(f"Base   -> F1: {met_base['f1']:.4f} | ROC-AUC: {met_base['roc_auc']:.4f}")
    print("Artefactos guardados en:")
    for nombre in ("modelo.joblib", "metrics.json", "metadata.json"):
        print(f"  {os.path.join(dir_version, nombre)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)