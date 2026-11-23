"""Pruebas del registro de solicitudes a /predict (Fase 4)."""
import csv
import os
import re

import api


def leer_registro():
    with open(api.LOG_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_solicitud_valida_deja_una_fila(cliente, registro_valido):
    assert cliente.post("/predict", json=registro_valido).status_code == 200
    filas = leer_registro()
    assert len(filas) == 1
    assert filas[0]["estado_http"] == "200"
    assert filas[0]["version"] == "v1"
    assert filas[0]["prediccion"] in ("Yes", "No")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", filas[0]["fecha_hora"])


def test_solicitud_rechazada_por_validacion_deja_una_fila(cliente, registro_valido):
    registro_valido.pop("tenure")
    assert cliente.post("/predict", json=registro_valido).status_code == 422
    filas = leer_registro()
    assert len(filas) == 1
    assert filas[0]["estado_http"] == "422"
    assert filas[0]["version"] == "" and filas[0]["prediccion"] == ""


def test_dos_validas_y_una_invalida_dejan_tres_filas(cliente, registro_valido):
    invalido = dict(registro_valido)
    invalido.pop("tenure")
    for cuerpo in (registro_valido, registro_valido, invalido):
        cliente.post("/predict", json=cuerpo)
    assert [f["estado_http"] for f in leer_registro()] == ["200", "200", "422"]


def test_modelo_no_disponible_registra_503_sin_version(cliente, registro_valido, monkeypatch):
    monkeypatch.setattr(api, "VERSION_ACTIVA", "v-no-existe")
    assert cliente.post("/predict", json=registro_valido).status_code == 503
    filas = leer_registro()
    assert filas[0]["estado_http"] == "503" and filas[0]["version"] == ""


def test_otras_rutas_no_se_registran(cliente):
    cliente.get("/health")
    cliente.post("/train", json={"version": "v 1"})        # 422 de /train
    assert not os.path.exists(api.LOG_PATH)


def test_el_registro_conserva_el_historial(cliente, registro_valido):
    os.makedirs(os.path.dirname(api.LOG_PATH))
    with open(api.LOG_PATH, "w", encoding="utf-8", newline="") as f:
        f.write("fecha_hora,estado_http,version,prediccion\n"
                "2026-11-06T10:00:00,200,v1,No\n")
    cliente.post("/predict", json=registro_valido)
    with open(api.LOG_PATH, encoding="utf-8") as f:
        texto = f.read()
    assert texto.count("fecha_hora") == 1                   # un solo encabezado
    assert len(leer_registro()) == 2 and "2026-11-06T10:00:00" in texto