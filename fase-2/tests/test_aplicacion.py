"""Pruebas de la aplicacion de churn (Fase 2).

Ejecutar desde la carpeta fase-2 con:  python -m pytest
Usan una muestra pequena de los datos locales (sin descargas) y carpetas
temporales (tmp_path), de modo que no tocan los modelos de la entrega.
"""
import json
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd

FASE2 = Path(__file__).resolve().parent.parent
CSV_REAL = FASE2 / "data" / "telco_churn.csv"


def _run(args):
    """Ejecuta un script de la fase-2 y devuelve el proceso terminado."""
    return subprocess.run([sys.executable] + args, cwd=FASE2,
                          capture_output=True, text=True)


def _datos_pequenos(tmp_path):
    """Muestra pequena y estratificada de los datos reales (sin descargas)."""
    df = pd.read_csv(CSV_REAL)
    muestra = df.groupby("Churn", group_keys=False).sample(n=30, random_state=0)
    ruta = tmp_path / "mini.csv"
    muestra.to_csv(ruta, index=False)
    return ruta


def _entrenar(tmp_path, version="v1", seed=42):
    out = tmp_path / "models"
    datos = _datos_pequenos(tmp_path)
    r = _run(["train.py", "--data", str(datos), "--version", version,
              "--output-dir", str(out), "--cv-folds", "2", "--seed", str(seed)])
    return out, datos, r


def test_entrenamiento_y_almacenamiento(tmp_path):
    out, _, r = _entrenar(tmp_path)
    assert r.returncode == 0, r.stderr
    dv = out / "v1"
    assert (dv / "modelo.joblib").exists()
    assert (dv / "metrics.json").exists()
    assert (dv / "metadata.json").exists()
    meta = json.loads((dv / "metadata.json").read_text(encoding="utf-8"))
    for clave in ("version", "algoritmo", "variable_objetivo",
                  "variables_entrada", "semilla", "particion", "python"):
        assert clave in meta
    assert hasattr(joblib.load(dv / "modelo.joblib"), "predict")


def test_carga_y_prediccion(tmp_path):
    out, datos, r = _entrenar(tmp_path)
    assert r.returncode == 0, r.stderr
    entrada = tmp_path / "entrada.csv"
    pd.read_csv(datos).drop(columns=["Churn"]).to_csv(entrada, index=False)
    salida = tmp_path / "pred.csv"
    rp = _run(["predict.py", "--model", str(out / "v1"),
               "--input", str(entrada), "--output", str(salida)])
    assert rp.returncode == 0, rp.stderr
    res = pd.read_csv(salida)
    assert "prediccion" in res.columns
    assert len(res) == len(pd.read_csv(entrada))          # una prediccion por fila
    # el artefacto guardado conserva los resultados del modelo
    sys.path.insert(0, str(FASE2))
    from src.datos import seleccionar_features
    modelo = joblib.load(out / "v1" / "modelo.joblib")
    esperado = modelo.predict(seleccionar_features(pd.read_csv(entrada)))
    assert (res["prediccion"].to_numpy() == esperado).all()


def test_valores_faltantes(tmp_path):
    out, datos, _ = _entrenar(tmp_path)
    entrada = pd.read_csv(datos).drop(columns=["Churn"]).astype({"TotalCharges": "object"})
    entrada.loc[entrada.index[0], "TotalCharges"] = ""    # celda nula admitida
    ruta_in = tmp_path / "con_nulo.csv"
    entrada.to_csv(ruta_in, index=False)
    salida = tmp_path / "pred.csv"
    rp = _run(["predict.py", "--model", str(out / "v1"),
               "--input", str(ruta_in), "--output", str(salida)])
    assert rp.returncode == 0, rp.stderr
    res = pd.read_csv(salida)
    assert res["prediccion"].notna().all()                # tambien predice la fila con nulo


def test_entrada_invalida_columna_ausente(tmp_path):
    out, datos, _ = _entrenar(tmp_path)
    entrada = pd.read_csv(datos).drop(columns=["Churn", "Contract"])
    ruta_in = tmp_path / "malo.csv"
    entrada.to_csv(ruta_in, index=False)
    salida = tmp_path / "pred.csv"
    rp = _run(["predict.py", "--model", str(out / "v1"),
               "--input", str(ruta_in), "--output", str(salida)])
    assert rp.returncode != 0                             # error con codigo distinto de cero
    assert not salida.exists()                            # no genera salida aparentemente valida


def test_versionamiento(tmp_path):
    out, datos, r1 = _entrenar(tmp_path, version="v1")
    assert r1.returncode == 0, r1.stderr
    r2 = _run(["train.py", "--data", str(datos), "--version", "v2",
               "--output-dir", str(out), "--cv-folds", "2", "--seed", "7"])
    assert r2.returncode == 0, r2.stderr
    assert (out / "v1" / "modelo.joblib").exists()        # la primera version se conserva
    assert (out / "v2" / "modelo.joblib").exists()
    # reusar un identificador existente no sobrescribe en silencio
    r3 = _run(["train.py", "--data", str(datos), "--version", "v1",
               "--output-dir", str(out), "--cv-folds", "2"])
    assert r3.returncode != 0