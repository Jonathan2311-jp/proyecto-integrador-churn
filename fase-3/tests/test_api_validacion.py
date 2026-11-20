"""Pruebas de validacion de entradas de la API (Fase 3)."""


def test_campo_obligatorio_ausente_se_rechaza(cliente, registro_valido):
    registro_valido.pop("tenure")
    r = cliente.post("/predict", json=registro_valido)
    assert r.status_code == 422
    assert "tenure" in str(r.json()["detail"])      # el mensaje nombra el campo


def test_tipo_incompatible_se_rechaza(cliente, registro_valido):
    registro_valido["tenure"] = "abc"
    r = cliente.post("/predict", json=registro_valido)
    assert r.status_code == 422
    assert "tenure" in str(r.json()["detail"])


def test_train_rechaza_identificador_invalido(cliente, tmp_path):
    for malo in ["../escapa", "v 1", "v1\n"]:
        r = cliente.post("/train", json={"version": malo})
        assert r.status_code == 422
    # no se creo ninguna carpeta fuera de models/
    assert sorted(p.name for p in tmp_path.iterdir()) == ["models"]


def test_train_no_sobrescribe_version_existente(cliente, tmp_path):
    metricas = tmp_path / "models" / "v1" / "metrics.json"
    antes = metricas.read_bytes()
    r = cliente.post("/train", json={"version": "v1"})
    assert r.status_code == 409
    assert metricas.read_bytes() == antes