# Proyecto Integrador — Fase 4: Monitoreo de la API

**Curso:** Modelos y Simulación de Sistemas I — 2026-II
**Docente:** Andrés Parra
**Integrantes:** Zareth Mariana Vega Sánchez · Jonathan Alfonso Pérez

Esta fase agrega el **registro de las solicitudes** a la API de la Fase 3 y un
**script de monitoreo** que mide cómo se usa el servicio (no la calidad del
modelo). Conserva `fase-3/` y todo su funcionamiento.

## 1. Qué se agregó

- `src/registro.py` — escribe una fila por cada solicitud a `/predict` (también
  las rechazadas por validación) en un archivo CSV.
- En `api.py`, un *middleware* que registra cada respuesta de `/predict`.
- `monitoreo.py` — lee el registro y muestra el total de solicitudes y el
  porcentaje de fallidas (respuestas HTTP 4xx o 5xx).

## 2. Entorno

Python 3.13.7. Desde la carpeta `fase-4`:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Arrancar la API con el registro

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

El registro se activa solo y se guarda en **`logs/solicitudes.csv`** (ruta
configurable con la variable `LOG_PATH`). El archivo se abre en modo de agregar,
así conserva el historial entre ejecuciones. Campos: `fecha_hora`,
`estado_http`, `version`, `prediccion`.

En Docker, el registro se guarda en una carpeta montada para que persista fuera
del contenedor:

```bash
docker build -t monitoreo-fase4:1.0 .
docker run --rm -p 8000:8000 -v "%cd%\models:/app/models:ro" -v "%cd%\data:/app/data:ro" -v "%cd%\logs:/app/logs" monitoreo-fase4:1.0
```

## 4. Versiones del modelo y cómo cambiar de versión

Se conservan dos versiones: **v1** (activa por defecto) y **v2**. La API predice
con la versión indicada en la variable `VERSION_ACTIVA` (por defecto `v1`). Para
volver a la anterior o cambiar de versión, se arranca la API con esa variable:

```bash
set VERSION_ACTIVA=v2
python -m uvicorn api:app --port 8000
```

## 5. Monitoreo

Con el servidor apagado (o en otra terminal), sobre el registro:

```bash
python monitoreo.py
```

Muestra el total de solicitudes y el porcentaje de fallidas. Si el registro está
vacío o no existe, informa total 0 y «sin solicitudes».

## 6. Pruebas

```bash
python -m pytest
```

Incluye las pruebas de la API (Fase 3) más las del registro y el monitoreo.

## 7. Reporte

Para esta entrega se arrancó la API con el registro activo y se enviaron **tres
solicitudes a `/predict`**: dos válidas y una con un campo obligatorio ausente
(se omitió `Contract`).

Muestra del registro (`logs/solicitudes.csv`):

```
fecha_hora,estado_http,version,prediccion
2026-11-23T01:54:00,200,v1,Yes
2026-11-23T01:54:05,200,v1,No
2026-11-23T01:54:10,422,,
```

Indicadores (`monitoreo.py`):

- **Total de solicitudes:** 3
- **Solicitudes fallidas (4xx/5xx):** 1
- **Porcentaje de fallos:** 33,3 %

La API procesó las dos solicitudes válidas con la versión `v1` y rechazó la
tercera por faltarle la variable `Contract`. Ese rechazo fue el **resultado
esperado de la prueba de validación**, no una falla del servicio; el porcentaje
describe el funcionamiento operativo, no la calidad predictiva del modelo.

### Política de reentrenamiento

Se revisará el modelo cuando se reúnan **100 nuevas solicitudes con resultado
real confirmado** (si el cliente efectivamente canceló o no). En ese momento, el
integrante responsable del modelo ejecutará `POST /train` para generar una
versión candidata, comparará la candidata con la versión activa usando **F1
sobre los mismos datos reservados**, y activará la candidata (cambiando
`VERSION_ACTIVA`) **solo si supera a la versión actual**.