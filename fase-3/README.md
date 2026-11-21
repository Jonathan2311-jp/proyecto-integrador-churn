# Fase 3 — API REST de predicción de churn

API construida con **FastAPI** sobre el trabajo de la Fase 2: reutiliza el
preprocesamiento y las versiones del modelo guardadas en `models/` y expone tres
operaciones por HTTP. El problema, los datos y el modelo (regresión logística)
son los de las fases anteriores.

## 1. Contenido de `fase-3/`

| Archivo | Función |
|---|---|
| `api.py` | Aplicación: `GET /health`, `POST /predict` y `POST /train` |
| `src/prediccion.py` | Carga una versión y predice un registro |
| `src/entrenamiento.py` | Entrena y guarda una versión (modelo, métricas y metadatos) |
| `src/datos.py`, `src/modelo.py`, `src/preprocesamiento.py` | Idénticos a los de la Fase 2 |
| `cliente.py` | Cliente de prueba: envía un registro a `/predict` |
| `tests/` | 8 pruebas con `pytest` |
| `Dockerfile`, `.dockerignore` | Imagen de la API |
| `models/` | Versiones `v1` y `v2`; las que crea `/train` se guardan aquí y no se versionan |
| `data/` | `telco_churn.csv` (entrenamiento) y `ejemplo_prediccion.csv` |

## 2. Entorno

Python 3.13.x (verificado con 3.13.5 y 3.13.7). Desde `fase-3`:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

La documentación interactiva queda en `http://localhost:8000/docs`.

## 3. Configuración y selección de versión

La API lee tres variables de entorno:

| Variable | Valor por defecto | Uso |
|---|---|---|
| `VERSION_ACTIVA` | `v1` | Versión que usa `/predict` |
| `MODELS_DIR` | `models` | Carpeta con las versiones |
| `DATA_PATH` | `data/telco_churn.csv` | Datos que usa `/train` |

Para usar otra versión disponible (`v1`, `v2` o una creada con `/train`):
PowerShell `$env:VERSION_ACTIVA="v2"` antes de arrancar, o `-e VERSION_ACTIVA=v2`
en Docker. La versión se lee al iniciar: cambiarla exige reiniciar la API.
`/train` guarda una versión nueva pero **no la activa**.

## 4. Operaciones

| Operación | Descripción | Respuesta |
|---|---|---|
| `GET /health` | Estado del servicio | 200 `{"status":"ok","version_activa":"v1"}` |
| `POST /predict` | Predice un registro con la versión activa | 200 con predicción y versión |
| `POST /train` | Entrena y guarda una versión nueva; el cuerpo puede ir vacío o traer `version` y `seed` | 201 con la versión creada y sus métricas |

### Esquema de `/predict`

Las 19 variables del proyecto, con los nombres y tipos del entrenamiento:
`tenure` (entero), `MonthlyCharges` (decimal), `TotalCharges` (decimal o `null`),
`SeniorCitizen` (solo `0` o `1`) y 15 variables de texto: `gender`, `Partner`,
`Dependents`, `PhoneService`, `MultipleLines`, `InternetService`,
`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`,
`StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling` y `PaymentMethod`.

- `TotalCharges: null` se trata con la regla guardada de la Fase 2 (se imputa con 0).
- Los campos adicionales se ignoran y una categoría no vista en el entrenamiento
  no produce error (se codifica con ceros, como en la Fase 2).
- La variable objetivo (`Churn`) no se envía.

### Ejemplo de solicitud y respuesta

```json
{"tenure": 1, "MonthlyCharges": 29.85, "TotalCharges": 29.85, "SeniorCitizen": 0,
 "gender": "Female", "Partner": "Yes", "Dependents": "No", "PhoneService": "No",
 "MultipleLines": "No phone service", "InternetService": "DSL",
 "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No",
 "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
 "Contract": "Month-to-month", "PaperlessBilling": "Yes",
 "PaymentMethod": "Electronic check"}
```

```json
{"prediccion": "Yes", "probabilidad_churn": 0.786274, "version": "v1"}
```

`"Yes"` significa que el cliente cancelaría el servicio; `"No"`, que permanecería.

### Errores

| Situación | Código |
|---|---|
| Campo obligatorio ausente, tipo incompatible o `SeniorCitizen` fuera de 0/1 | 422, con el nombre del campo en `loc` |
| La versión activa no existe en `MODELS_DIR` | 503, con el aviso de configurar `VERSION_ACTIVA` |
| `/train` con identificador inválido (solo letras, números, `.`, `_`, `-`) | 422 |
| `/train` con una versión que ya existe | 409 (no se sobrescribe) |
| `/train` sin el archivo de datos | 500 |

## 5. Docker

Imagen basada en `python:3.13-slim`. Los datos y los modelos **no** van dentro de la
imagen: entran por montaje, así lo que genera `/train` queda en el anfitrión aunque
el contenedor se elimine. Puerto: **8000**.

**PowerShell** (desde `fase-3`):

```powershell
docker build -t modelos-fase3:1.0 .
docker run --rm -p 8000:8000 -v "${PWD}\data:/app/data:ro" -v "${PWD}\models:/app/models" modelos-fase3:1.0
```

**CMD:** igual, escribiendo `%cd%` en lugar de `${PWD}`.

Para otra versión: `-e VERSION_ACTIVA=v2`.

### Evidencia

Construcción y arranque:

```text
<pega los pasos [1/6] a [6/6] y la línea "naming to">
<pega la línea "Uvicorn running on http://0.0.0.0:8000">
```

`/health`, `/predict` y rechazo de un registro sin `tenure`:

```text
[+] Building 87.8s (11/11) FINISHED      
 => [1/6] FROM docker.io/library/python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90                                          0.1s 
 => => resolve docker.io/library/python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90                                          0.1s
 => CACHED [2/6] WORKDIR /app                                                                                                                                      0.0s
 => [3/6] COPY requirements.txt .                                                                                                                                  0.1s 
 => [4/6] RUN pip install --no-cache-dir -r requirements.txt                                                                                                      53.4s 
 => [5/6] COPY src/ ./src/                                                                                                                                         0.3s 
 => [6/6] COPY api.py ./  
  => => naming to docker.io/library/modelos-fase3:1.0                                                                                                               0.0s 
 => => unpacking to docker.io/library/modelos-fase3:1.0  
 
 Respuesta  HTTP 200:
{
  "prediccion": "Yes",
  "probabilidad_churn": 0.786274,
  "version": "v1"
}
 
s:/app/models" modelos-fase3:1.0                                                                                                                                        
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

`/train` y persistencia (la versión sigue en `models/` tras detener el contenedor):

```text

version                   f1 roc_auc mensaje
-------                   -- ------- -------
v-api-20261121-234222 0,6181  0,8413 Version 'v-api-20261121-234222' entrenada y guardada.


Mode                 LastWriteTime         Length Name                                                                                                                 
----                 -------------         ------ ----                                                                                                                 
d-----     18/11/2026  10:43 p. m.                v-api-20261119-034324                                                                                                
d-----      21/11/2026  6:42 p. m.                v-api-20261121-234222                                                                                                
d-----      18/11/2026  9:03 p. m.                v1                                                                                                                   
d-----      18/11/2026  9:03 p. m.                v2                                                                                                                   

```

Otra versión y modelo no disponible:

```text
version                   f1 roc_auc mensaje
-------                   -- ------- -------
v-api-20261121-234517 0,6181  0,8413 Version 'v-api-20261121-234517' entrenada y guardada.

Respuesta  HTTP 200:
{
  "prediccion": "Yes",
  "probabilidad_churn": 0.822859,
  "version": "v2"
}

Mode                 LastWriteTime         Length Name                                                                                                                 
----                 -------------         ------ ----                                                                                                                 
d-----     18/11/2026  10:43 p. m.                v-api-20261119-034324                                                                                                
d-----      21/11/2026  6:42 p. m.                v-api-20261121-234222                                                                                                
d-----      21/11/2026  6:45 p. m.                v-api-20261121-234517                                                                                                
d-----      18/11/2026  9:03 p. m.                v1                                                                                                                   
d-----      18/11/2026  9:03 p. m.                v2                                                                                                                   

```

## 6. Cliente de prueba

`cliente.py` lee una fila de `data/ejemplo_prediccion.csv`, la convierte a JSON y
la envía a `/predict`. Desde `fase-3`, con la API en marcha:

```powershell
python cliente.py                  # primera fila
python cliente.py --fila 2         # otra fila
python cliente.py --url http://localhost:8000
```

Termina con código 1 si la API no responde o la respuesta no es 200.

## 7. Pruebas

```powershell
python -m pytest -v
```

8 pruebas, sin entrenar ni usar Docker (copian `v1` a una carpeta temporal):

| Archivo | Comprueba |
|---|---|
| `test_api_basico.py` | `/health` da 200; un registro válido devuelve predicción y versión; `TotalCharges` nulo da lo mismo que 0 |
| `test_api_validacion.py` | Campo ausente, tipo incompatible y `SeniorCitizen` fuera de 0/1 se rechazan con 422; `/train` rechaza identificadores inválidos y no sobrescribe versiones |

`/train` se comprobó manualmente (ver la evidencia de la sección 5).

## 8. Limitaciones

- `/train` es síncrono: la respuesta llega cuando termina el entrenamiento.
- La versión activa se fija al arrancar y `/train` no la cambia.
- El modelo se carga desde disco en cada `/predict`.
- Dentro del contenedor, el campo `commit` de `metadata.json` queda como
  `desconocido` porque la imagen no incluye `.git`.
- La API no tiene autenticación.