"""Fixtures compartidas de las pruebas de la API (Fase 3)."""
import shutil
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

FASE3 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FASE3))

import api  # noqa: E402


@pytest.fixture()
def cliente(tmp_path, monkeypatch):
    """Cliente de la API con una copia temporal de v1 (no toca models/)."""
    shutil.copytree(FASE3 / "models" / "v1", tmp_path / "models" / "v1")
    monkeypatch.setattr(api, "MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(api, "VERSION_ACTIVA", "v1")
    return TestClient(api.app)


@pytest.fixture()
def registro_valido():
    """Un cliente completo con las 19 variables predictoras."""
    return {
        "tenure": 1, "MonthlyCharges": 29.85, "TotalCharges": 29.85,
        "SeniorCitizen": 0, "gender": "Female", "Partner": "Yes",
        "Dependents": "No", "PhoneService": "No",
        "MultipleLines": "No phone service", "InternetService": "DSL",
        "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    }