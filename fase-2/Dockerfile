# Imagen base con version explicita de Python (no se usa 'latest')
FROM python:3.13-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar dependencias primero, para aprovechar la cache de capas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el codigo de la aplicacion.
# Los datos, modelos y resultados NO se copian: entran por montaje al ejecutar.
COPY src/ ./src/
COPY train.py predict.py ./

# No se entrena durante el build. Comando por defecto: mostrar la ayuda.
CMD ["python", "train.py", "--help"]