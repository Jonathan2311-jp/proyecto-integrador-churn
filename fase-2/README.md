# Proyecto Integrador — Fase 2: Predicción de fuga de clientes (churn)

**Curso:** Modelos y Simulación de Sistemas I — 2026-II
**Docente:** Andrés Parra
**Integrantes:** Zareth Mariana Vega Sánchez · Jonathan Alfonso Pérez

Aplicación reutilizable que entrena un modelo de predicción de cancelación
(*churn*) de clientes de telecomunicaciones y genera predicciones mediante
comandos, sin ejecutar el notebook.

## 1. Problema y relación con la Fase 1

Una empresa de telecomunicaciones necesita anticipar qué clientes cancelarán
el servicio para focalizar la retención. Es un problema de **clasificación
binaria** (variable objetivo `Churn`: `Yes`/`No`), con clases desbalanceadas
(~26,5 % / 73,5 %).

La Fase 1 (notebook) eligió el problema, exploró los datos, entrenó el modelo
y lo guardó. La Fase 2 **conserva el mismo problema, datos y enfoque** y
convierte ese trabajo en una aplicación con scripts `train.py` / `predict.py`,
preprocesamiento reutilizable, versionamiento, pruebas y Docker. El notebook
original se conserva en la carpeta `fase-1/`.

## 2. Correcciones respecto a la Fase 1

- **Preprocesamiento autosuficiente.** En la Fase 1, la conversión de
  `TotalCharges` de texto a número y el relleno de sus nulos con 0 se hacían
  **fuera** del modelo (en el notebook), de modo que el `.joblib` dependía de
  pasos externos. En la Fase 2 esa limpieza determinista viaja **dentro** del
  `Pipeline` (paso `FunctionTransformer`), así el artefacto se carga y predice
  sobre datos crudos sin repetir transformaciones.
- **Modularización.** La lógica se reorganizó en el paquete `src/`
  (`datos`, `preprocesamiento`, `modelo`), reutilizado por ambos scripts.
- **Línea de comandos y versionamiento.** Entradas por `argparse`, versiones
  identificables del modelo y metadatos de cada ejecución.

## 3. Entorno y dependencias

- **Python 3.13.7**
- Crear el entorno e instalar dependencias (desde la carpeta `fase-2`):

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows CMD/PowerShell
pip install -r requirements.txt
```

## 4. Datos

- **Fuente:** conjunto *Telco Customer Churn* (IBM).
- **Ubicación:** `fase-2/data/telco_churn.csv` (incluido en el repositorio;
  copia del usado en la Fase 1, en `fase-1/data/`).
- **Esquema del CSV de predicción:** las 19 variables de entrada, **sin** la
  columna objetivo `Churn`. Se incluye un ejemplo en
  `data/ejemplo_prediccion.csv`. Columnas requeridas: `tenure`,
  `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`, `gender`, `Partner`,
  `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`,
  `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`,
  `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`,
  `PaymentMethod`. La columna `customerID` es opcional (se ignora).

## 5. Entrenamiento (`train.py`)

```bash
python train.py --data data/telco_churn.csv --target Churn --output-dir models --version v1 --seed 42
python train.py --help
```

Lee y valida los datos, separa entrenamiento/prueba (estratificado, 80/20),
ajusta el preprocesamiento solo con entrenamiento, selecciona `C` por
validación cruzada de 5 particiones, evalúa en prueba y contra un modelo base,
y guarda la versión.

## 6. Predicción (`predict.py`)

```bash
python predict.py --model models/v1 --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
python predict.py --help
```

Carga una versión guardada (acepta la carpeta `models/v1` o la ruta directa al
`.joblib`) y escribe un CSV que conserva las filas de entrada más las columnas
`prediccion` (1 = Cancela / `Churn=Yes`, 0 = No cancela),
`prediccion_etiqueta` y `probabilidad_churn`. Si falta una columna requerida o
el modelo no existe, termina con un error claro y código de salida distinto de
cero.

## 7. Versiones, métricas y metadatos

Cada ejecución de `train.py` crea `models/<version>/` con:
- `modelo.joblib` — pipeline reutilizable (preprocesamiento + modelo),
- `metrics.json` — métricas del modelo y del modelo base,
- `metadata.json` — versión, fecha, algoritmo, variables, semilla, partición,
  versión de Python y hash del commit.

Se entregan dos versiones: **v1** (semilla 42) y **v2** (semilla 7). Reutilizar
un identificador existente no sobrescribe: el script termina con error.

## 8. Docker

Terminal usada por el equipo: **CMD de Windows** (las rutas de montaje usan
`%cd%`; en PowerShell se reemplaza por `${PWD}`, en bash por `$PWD`).

```bat
docker build -t modelos-fase2:1.0 .

docker run --rm -v "%cd%\data:/app/data:ro" -v "%cd%\models:/app/models" modelos-fase2:1.0 python train.py --data data/telco_churn.csv --target Churn --output-dir models --version v-docker --seed 42

docker run --rm -v "%cd%\data:/app/data:ro" -v "%cd%\models:/app/models:ro" -v "%cd%\outputs:/app/outputs" modelos-fase2:1.0 python predict.py --model models/v-docker/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
```

La imagen no se entrena en el *build*; los datos entran por montaje y los
modelos y resultados persisten fuera del contenedor.

## 9. Pruebas (`pytest`)

```bash
python -m pytest -v
```

Cinco pruebas en `tests/test_aplicacion.py`: (1) entrenamiento y almacenamiento
generan artefacto, métricas y metadatos; (2) carga y predicción dan una
predicción por fila y conservan los resultados; (3) una celda nula en
`TotalCharges` se procesa; (4) una columna faltante produce error con código
distinto de cero; (5) el versionamiento conserva la versión previa y no
sobrescribe. Usan una muestra pequeña local y carpetas temporales.

## 10. Resultados y comparación con el modelo base

| Métrica (prueba) | Modelo base | v1 (semilla 42) |
|---|---|---|
| Accuracy | 0.7346 | 0.7431 |
| Recall | 0.0000 | 0.7834 |
| F1 | 0.0000 | 0.6181 |
| ROC-AUC | 0.5000 | 0.8413 |

El modelo pasa de no detectar ninguna cancelación (modelo base) a detectar el
78 % de ellas. **v2** (semilla 7) obtuvo F1 0.6505 y ROC-AUC 0.8570.

**Limitaciones conocidas:**
- v1 y v2 usan semillas distintas, por lo que la **partición de prueba cambia**
  y sus métricas no son directamente comparables entre sí; cada una se compara
  contra su propio modelo base.
- Solo se define tratamiento de nulos para `TotalCharges` (relleno con 0 por
  su significado); el resto de columnas se asumen completas.
- Categorías no vistas en predicción se ignoran (`OneHotEncoder` con
  `handle_unknown='ignore'`); columnas adicionales se descartan.

## 11. Fuentes

- Conjunto de datos *Telco Customer Churn* (IBM):
  https://github.com/IBM/telco-customer-churn-on-icp4d

## 12. Control de versiones (Git)

Flujo: `main` (estable), `develop` (integración), `feature/*` (tareas). La
Entrega 2 se integra en `main` y se identifica con la etiqueta `entrega-2`.