"""Pruebas basicas de la API: /health y /predict valido (Fase 3)."""


def test_health_responde_200(cliente):
    r = cliente.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_valido_devuelve_prediccion_y_version(cliente, registro_valido):
    r = cliente.post("/predict", json=registro_valido)
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["prediccion"] in ("Yes", "No")
    assert 0.0 <= cuerpo["probabilidad_churn"] <= 1.0
    assert cuerpo["version"] == "v1"


def test_totalcharges_nulo_aplica_la_regla_guardada(cliente, registro_valido):
    # Regla de la Fase 2: un TotalCharges nulo se imputa con 0.
    registro_valido["TotalCharges"] = None
    r = cliente.post("/predict", json=registro_valido)
    assert r.status_code == 200
    con_cero = dict(registro_valido, TotalCharges=0.0)
    esperado = cliente.post("/predict", json=con_cero).json()["probabilidad_churn"]
    assert r.json()["probabilidad_churn"] == esperado