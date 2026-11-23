"""Pruebas del script de monitoreo (Fase 4)."""
import subprocess
import sys
from pathlib import Path

import monitoreo

FASE4 = Path(__file__).resolve().parent.parent
ENCABEZADO = "fecha_hora,estado_http,version,prediccion\n"


def ejecutar(ruta):
    return subprocess.run([sys.executable, "monitoreo.py", "--registro", str(ruta)],
                          cwd=FASE4, capture_output=True, text=True)


def test_tres_solicitudes_una_fallida():
    total, fallidas, porcentaje = monitoreo.calcular([200, 200, 422])
    assert (total, fallidas) == (3, 1)
    assert round(porcentaje, 1) == 33.3


def test_solo_4xx_y_5xx_cuentan_como_fallidas():
    assert monitoreo.calcular([200, 399, 400, 499, 500, 503, 599])[1] == 5


def test_script_muestra_total_y_porcentaje(tmp_path):
    registro = tmp_path / "registro.csv"
    registro.write_text(ENCABEZADO
                        + "2026-11-06T10:00:00,200,v1,Yes\n"
                        + "2026-11-06T10:01:00,200,v1,No\n"
                        + "2026-11-06T10:02:00,422,,\n", encoding="utf-8")
    r = ejecutar(registro)
    assert r.returncode == 0
    assert "Total de solicitudes: 3" in r.stdout
    assert "Porcentaje de fallos: 33,3 %" in r.stdout


def test_registro_vacio_indica_sin_solicitudes(tmp_path):
    for contenido in (ENCABEZADO, ""):                       # solo encabezado y archivo vacio
        registro = tmp_path / "vacio.csv"
        registro.write_text(contenido, encoding="utf-8")
        r = ejecutar(registro)
        assert r.returncode == 0
        assert "Total de solicitudes: 0" in r.stdout
        assert "sin solicitudes" in r.stdout


def test_estado_invalido_termina_con_error(tmp_path):
    registro = tmp_path / "malo.csv"
    registro.write_text(ENCABEZADO + "2026-11-06T10:00:00,abc,v1,Yes\n", encoding="utf-8")
    r = ejecutar(registro)
    assert r.returncode != 0 and "estado_http" in r.stderr