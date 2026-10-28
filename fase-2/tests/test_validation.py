"""Pruebas adicionales de validacion de entrada y trazabilidad (Fase 2)."""
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

FASE2 = Path(__file__).resolve().parent.parent
CSV_REAL = FASE2 / "data" / "telco_churn.csv"


def _run(args):
    return subprocess.run([sys.executable] + args, cwd=FASE2,
                          capture_output=True, text=True)


@pytest.fixture(scope="module")
def entorno(tmp_path_factory):
    """Entrena una sola vez con una muestra pequena y reutiliza el modelo."""
    base = tmp_path_factory.mktemp("validacion")
    df = pd.read_csv(CSV_REAL)
    mini = df.groupby("Churn", group_keys=False).sample(n=30, random_state=0)
    datos = base / "mini.csv"
    mini.to_csv(datos, index=False)
    r = _run(["train.py", "--data", str(datos), "--version", "v1",
              "--output-dir", str(base / "models"), "--cv-folds", "2"])
    assert r.returncode == 0, r.stderr
    entrada = mini.drop(columns=["Churn"]).reset_index(drop=True)
    return base, base / "models" / "v1", entrada


def _predecir(base, modelo, entrada, nombre):
    ruta_in, ruta_out = base / f"{nombre}_in.csv", base / f"{nombre}_out.csv"
    entrada.to_csv(ruta_in, index=False)
    r = _run(["predict.py", "--model", str(modelo),
              "--input", str(ruta_in), "--output", str(ruta_out)])
    return r, ruta_out


def test_metricas_modelo_y_base(entorno):
    _, version, _ = entorno
    m = json.loads((version / "metrics.json").read_text(encoding="utf-8"))
    for clave in ("modelo", "modelo_base"):
        assert set(m[clave]) >= {"accuracy", "precision", "recall", "f1", "roc_auc"}
        assert all(0.0 <= v <= 1.0 for v in m[clave].values())
    assert m["modelo_base"]["f1"] == 0.0      # el modelo base nunca predice churn


def test_probabilidades_coinciden_con_el_modelo(entorno):
    base, version, entrada = entorno
    r, salida = _predecir(base, version, entrada, "prob")
    assert r.returncode == 0, r.stderr
    res = pd.read_csv(salida)
    sys.path.insert(0, str(FASE2))
    from src.datos import seleccionar_features
    esperado = joblib.load(version / "modelo.joblib").predict_proba(
        seleccionar_features(entrada))[:, 1]
    assert np.allclose(res["probabilidad_churn"], esperado, atol=1e-6)


def test_orden_de_columnas_y_columnas_extra(entorno):
    base, version, entrada = entorno
    r0, s0 = _predecir(base, version, entrada, "orden0")
    alterada = entrada[list(reversed(entrada.columns))].copy()
    alterada["columna_extra"] = 1
    r1, s1 = _predecir(base, version, alterada, "orden1")
    assert r0.returncode == 0 and r1.returncode == 0
    a, b = pd.read_csv(s0), pd.read_csv(s1)
    assert len(a) == len(b) == len(entrada)               # no se pierden filas
    assert (a["prediccion"] == b["prediccion"]).all()     # el orden no afecta


def test_categoria_no_vista_no_falla(entorno):
    base, version, entrada = entorno
    nueva = entrada.copy()
    nueva.loc[0, "Contract"] = "Five year"
    r, salida = _predecir(base, version, nueva, "catnueva")
    assert r.returncode == 0, r.stderr
    assert len(pd.read_csv(salida)) == len(entrada)


def test_tipo_invalido_indica_columna_y_fila(entorno):
    base, version, entrada = entorno
    mala = entrada.copy()
    mala["tenure"] = mala["tenure"].astype(object)
    mala.loc[3, "tenure"] = "abc"
    r, salida = _predecir(base, version, mala, "tipo")
    assert r.returncode != 0
    assert "tenure" in r.stderr and "[5]" in r.stderr     # fila 3 -> fila 5 del CSV
    assert not salida.exists()


def test_nulo_en_variable_que_no_admite_nulos(entorno):
    base, version, entrada = entorno
    mala = entrada.copy()
    mala.loc[0, "MonthlyCharges"] = None
    r, salida = _predecir(base, version, mala, "nulo")
    assert r.returncode != 0
    assert "MonthlyCharges" in r.stderr and "nulos" in r.stderr
    assert not salida.exists()


def test_modelo_inexistente_termina_con_error(entorno):
    base, _, entrada = entorno
    r, salida = _predecir(base, base / "models" / "no_existe", entrada, "sinmodelo")
    assert r.returncode != 0
    assert "No se encontro el modelo" in r.stderr
    assert not salida.exists()