# Imagen base con version explicita de Python (no se usa 'latest')
FROM python:3.13-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar dependencias primero, para aprovechar la cache de capas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Codigo de la API. Los datos y los modelos NO se copian: entran por montaje.
COPY src/ ./src/
COPY api.py ./

# Configuracion por defecto (se puede cambiar con -e al ejecutar el contenedor)
ENV MODELS_DIR=models \
    DATA_PATH=data/telco_churn.csv \
    VERSION_ACTIVA=v1

# Puerto en el que escucha la API
EXPOSE 8000

# 0.0.0.0 para que la API sea accesible desde fuera del contenedor
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]