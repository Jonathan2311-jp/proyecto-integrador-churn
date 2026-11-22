"""API REST de prediccion de churn (Fase 3).

Expone tres operaciones sobre el modelo de la Fase 2, reutilizando su
preprocesamiento y sus funciones de entrenamiento y prediccion:

    GET  /health   -> estado del servicio
    POST /predict  -> prediccion de un registro
    POST /train    -> entrena y guarda una version nueva

Arranque:  uvicorn api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations
import os
import sys
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.datos import TARGET
from src.prediccion import cargar_modelo, predecir_registro
from fastapi import FastAPI, HTTPException, Request
from src.registro import registrar_solicitud
from src.entrenamiento import entrenar_y_guardar, validar_version

# Configuracion por variables de entorno (con valores por defecto)
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
DATA_PATH = os.environ.get("DATA_PATH", "data/telco_churn.csv")
VERSION_ACTIVA = os.environ.get("VERSION_ACTIVA", "v1")

LOG_PATH = os.environ.get("LOG_PATH", "logs/solicitudes.csv")

app = FastAPI(title="API de prediccion de churn", version="1.0")

@app.middleware("http")
async def registrar_predicciones(request: Request, call_next):
    """Deja una sola fila por cada solicitud a /predict, tambien las rechazadas.

    Se hace aqui (y no dentro de la ruta) porque la validacion de FastAPI puede
    rechazar la entrada antes de llamar a la funcion de prediccion.
    """
    respuesta = await call_next(request)
    if request.url.path == "/predict":
        try:
            registrar_solicitud(
                LOG_PATH, respuesta.status_code,
                getattr(request.state, "version", ""),
                getattr(request.state, "prediccion", ""))
        except OSError as e:   # un fallo del registro no debe tumbar la prediccion
            print(f"AVISO: no se pudo escribir el registro: {e}", file=sys.stderr)
    return respuesta

# ---- Esquema de entrada de /predict (las 19 variables del proyecto) ----
class ClienteChurn(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: Optional[float] = None   # admite nulo -> se rellena con 0
    SeniorCitizen: Literal[0, 1]
    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str


class EntrenarRequest(BaseModel):
    version: Optional[str] = None     # si no se da, se genera uno con la fecha
    seed: Optional[int] = 42


@app.get("/health")
def health():
    """Latido del servicio."""
    return {"status": "ok", "version_activa": VERSION_ACTIVA}


@app.post("/predict")
def predict(cliente: ClienteChurn, request: Request):
    """Predice un registro con la version configurada."""
    ruta = os.path.join(MODELS_DIR, VERSION_ACTIVA)
    try:
        modelo = cargar_modelo(ruta)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=(f"El modelo '{VERSION_ACTIVA}' no esta disponible. "
                    f"Configure la variable VERSION_ACTIVA a una version "
                    f"existente en '{MODELS_DIR}'."))
    try:
        resultado = predecir_registro(modelo, cliente.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    request.state.version = VERSION_ACTIVA  # para el registro (Fase 4)
    request.state.prediccion = resultado["prediccion"]
    return {
        "prediccion": resultado["prediccion"],      # "Yes" (cancela) / "No"
        "probabilidad_churn": resultado["probabilidad_churn"],
        "version": VERSION_ACTIVA,
    }


@app.post("/train", status_code=201)
def train(req: Optional[EntrenarRequest] = None):
    """Entrena y guarda una version nueva. Acepta cuerpo vacio."""
    req = req or EntrenarRequest()
    version = req.version or ("v-api-" +
                              datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    try:
        validar_version(version)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    try:
        resumen = entrenar_y_guardar(
            data=DATA_PATH, target=TARGET, output_dir=MODELS_DIR,
            version=version,
            seed=req.seed if req.seed is not None else 42)
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "version": resumen["version"],
        "f1": round(resumen["f1"], 4),
        "roc_auc": round(resumen["roc_auc"], 4),
        "mensaje": f"Version '{resumen['version']}' entrenada y guardada.",
    }