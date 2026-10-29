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

- **Python 3.13.x** (verificado con 3.13.5 y 3.13.7; la imagen Docker usa
  `python:3.13-slim`).
- Todos los comandos se ejecutan desde la carpeta `fase-2`.

**Windows PowerShell**

```powershell
cd fase-2
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Si PowerShell indica que la ejecución de scripts está deshabilitada, se
habilita una sola vez para el usuario actual:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

**Windows CMD**

```bat
cd fase-2
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

**Linux / macOS (bash)**

```bash
cd fase-2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Problemas frecuentes:
- `Could not open requirements file`: el comando se ejecutó fuera de `fase-2`.
- Un `.venv` con carpeta `bin` en lugar de `Scripts` se creó con un Python que
  no es el nativo de Windows; se borra y se recrea con `py -m venv .venv`.
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

Terminales verificadas: **Windows PowerShell** y **CMD de Windows**. Solo cambia cómo se escribe la ruta del directorio actual:
`${PWD}` en PowerShell, `%cd%` en CMD y `$PWD` en bash. Requiere Docker Desktop
en ejecución.

**PowerShell**

```powershell
docker build -t modelos-fase2:1.0 .
New-Item -ItemType Directory -Force models, outputs | Out-Null

docker run --rm -v "${PWD}\data:/app/data:ro" -v "${PWD}\models:/app/models" modelos-fase2:1.0 python train.py --data data/telco_churn.csv --target Churn --output-dir models --version v-docker --seed 42

docker run --rm -v "${PWD}\data:/app/data:ro" -v "${PWD}\models:/app/models:ro" -v "${PWD}\outputs:/app/outputs" modelos-fase2:1.0 python predict.py --model models/v-docker/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
```

**CMD**
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
  `handle_unknown='ignore'`); columnas adicionales se ignoran al predecir y se conservan en el CSV de salida.
- Dentro del contenedor, el campo `commit` de `metadata.json` queda como
  `desconocido` porque la imagen no incluye `.git`; `v1` y `v2` (entrenadas en
  local) sí guardan el hash del código.

## 11. Evidencia de ejecución

Salidas reales obtenidas por el equipo en Windows (PowerShell) con Python 3.13.5 y las dependencias de `requirements.txt`.
Todos los comandos se ejecutan desde la carpeta `fase-2`.

### 11.1 Pruebas

```text
> python -m pytest -v
collected 12 items

tests/test_aplicacion.py::test_entrenamiento_y_almacenamiento PASSED                                             [  8%]
tests/test_aplicacion.py::test_carga_y_prediccion PASSED                                                         [ 16%]
tests/test_aplicacion.py::test_valores_faltantes PASSED                                                          [ 25%]
tests/test_aplicacion.py::test_entrada_invalida_columna_ausente PASSED                                           [ 33%]
tests/test_aplicacion.py::test_versionamiento PASSED                                                             [ 41%]
tests/test_validation.py::test_metricas_modelo_y_base PASSED                                                     [ 50%]
tests/test_validation.py::test_probabilidades_coinciden_con_el_modelo PASSED                                     [ 58%]
tests/test_validation.py::test_orden_de_columnas_y_columnas_extra PASSED                                         [ 66%]
tests/test_validation.py::test_categoria_no_vista_no_falla PASSED                                                [ 75%]
tests/test_validation.py::test_tipo_invalido_indica_columna_y_fila PASSED                                        [ 83%]
tests/test_validation.py::test_nulo_en_variable_que_no_admite_nulos PASSED                                       [ 91%]
tests/test_validation.py::test_modelo_inexistente_termina_con_error PASSED                                       [100%]

============================================ 12 passed in 95.98s (0:01:35) ============================================
```

### 11.2 Entrenamiento y versionamiento

```text
> python train.py --data data/telco_churn.csv --target Churn --output-dir outputs/evidencia --version v1 --seed 42
=== Entrenamiento completado ===
Version:      v1
Mejor C (CV): 0.1  | F1 CV: 0.6299
Modelo -> F1: 0.6181 | ROC-AUC: 0.8413 | recall: 0.7834
Base   -> F1: 0.0000 | ROC-AUC: 0.5000
Artefactos guardados en:
  outputs/evidencia\v1\modelo.joblib
  outputs/evidencia\v1\metrics.json
  outputs/evidencia\v1\metadata.json
```

Reentrenar con la semilla 42 reproduce las métricas de la versión `v1`
guardada en `models/`. Reutilizar un identificador existente se rechaza sin
sobrescribir nada:

```text
> python train.py ... --version v1 --seed 42
ERROR: La version 'v1' ya existe en outputs/evidencia\v1. Use otro identificador para no sobrescribirla.
(código de salida: 1)
```

### 11.3 Predicción con las dos versiones

```text
> python predict.py --model models/v1 --input data/ejemplo_prediccion.csv --output outputs/pred_v1.csv
=== Prediccion completada ===
Modelo:  models/v1\modelo.joblib
Entrada: data/ejemplo_prediccion.csv  (8 filas)
Salida:  outputs/pred_v1.csv
Marcados como 'Cancela' (1): 6 de 8
Etiquetas: 1 = el cliente cancelaria (churn), 0 = permaneceria.

> python predict.py --model models/v2 --input data/ejemplo_prediccion.csv --output outputs/pred_v2.csv
=== Prediccion completada ===
Modelo:  models/v2\modelo.joblib
Entrada: data/ejemplo_prediccion.csv  (8 filas)
Salida:  outputs/pred_v2.csv
Marcados como 'Cancela' (1): 6 de 8
Etiquetas: 1 = el cliente cancelaria (churn), 0 = permaneceria.
```

Primeras filas de cada salida (la correspondencia con la entrada se conserva):

```text
> Import-Csv outputs\pred_v1.csv | Select-Object -First 3 customerID,prediccion,prediccion_etiqueta,probabilidad_churn | Format-Table

customerID prediccion prediccion_etiqueta probabilidad_churn
---------- ---------- ------------------- ------------------
7590-VHVEG 1          Cancela             0.786274
5575-GNVDE 0          No cancela          0.139354
3668-QPYBK 1          Cancela             0.528653

> Import-Csv outputs\pred_v2.csv | Select-Object -First 3 customerID,prediccion,prediccion_etiqueta,probabilidad_churn | Format-Table

customerID prediccion prediccion_etiqueta probabilidad_churn
---------- ---------- ------------------- ------------------
7590-VHVEG 1          Cancela             0.822859
5575-GNVDE 0          No cancela          0.107921
3668-QPYBK 1          Cancela             0.535908
```

### 11.4 Entrada inválida

Con la columna `tenure` conteniendo texto, el proceso termina con un error
comprensible y no genera archivo de salida:

```text
> python predict.py --model models/v1 --input outputs/malo.csv --output outputs/p_malo.csv
ERROR: Entrada invalida:
  - 'tenure' debe ser numerica; hay valores no numericos en las filas [2, 4] del CSV
(código de salida: 1)
```

### 11.5 Docker

Construcción de la imagen (se omiten las líneas de descarga y exportación de capas).
No se entrena durante el *build*:

```text
> docker build -t modelos-fase2:1.0 .
[+] Building 90.5s (11/11) FINISHED
 => [1/6] FROM docker.io/library/python:3.13-slim
 => [2/6] WORKDIR /app
 => [3/6] COPY requirements.txt .
 => [4/6] RUN pip install --no-cache-dir -r requirements.txt
 => [5/6] COPY src/ ./src/
 => [6/6] COPY train.py predict.py ./
 => => naming to docker.io/library/modelos-fase2:1.0

> docker images modelos-fase2
REPOSITORY      TAG       IMAGE ID       CREATED         SIZE
modelos-fase2   1.0       a19501cd0a69   2 minutes ago   680MB
```

Entrenamiento dentro del contenedor (datos montados en solo lectura, modelos
persistidos en el anfitrión):

```text
> docker run --rm -v "${PWD}\data:/app/data:ro" -v "${PWD}\models:/app/models" modelos-fase2:1.0 python train.py --data data/telco_churn.csv --target Churn --output-dir models --version v-docker --seed 42
=== Entrenamiento completado ===
Version:      v-docker
Mejor C (CV): 0.1  | F1 CV: 0.6299
Modelo -> F1: 0.6181 | ROC-AUC: 0.8413 | recall: 0.7834
Base   -> F1: 0.0000 | ROC-AUC: 0.5000
Artefactos guardados en:
  models/v-docker/modelo.joblib
  models/v-docker/metrics.json
  models/v-docker/metadata.json
```

Las métricas coinciden con las de `v1` entrenada en local. Predicción dentro del contenedor:

```text
> docker run --rm -v "${PWD}\data:/app/data:ro" -v "${PWD}\models:/app/models:ro" -v "${PWD}\outputs:/app/outputs" modelos-fase2:1.0 python predict.py --model models/v-docker/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
=== Prediccion completada ===
Modelo:  models/v-docker/modelo.joblib
Entrada: data/ejemplo_prediccion.csv  (8 filas)
Salida:  outputs/predicciones.csv
Marcados como 'Cancela' (1): 6 de 8
Etiquetas: 1 = el cliente cancelaria (churn), 0 = permaneceria.
```

Persistencia: con los contenedores ya eliminados (`--rm`), los resultados
siguen en el anfitrión (se omiten otros archivos locales de `outputs/`):

```text
> docker ps -a
CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES

> dir models\v-docker
Mode                 LastWriteTime         Length Name
-a----      28/10/2026  8:11 p. m.           1797 metadata.json
-a----      28/10/2026  8:11 p. m.            402 metrics.json
-a----      28/10/2026  8:11 p. m.          10417 modelo.joblib

> dir outputs\predicciones.csv
-a----      28/10/2026  8:12 p. m.           1371 predicciones.csv

> Select-String '"python"|"commit"' models\v-docker\metadata.json
"python": "3.13.18"
"commit": "desconocido"
```

El `commit` figura como `desconocido` porque la imagen no incluye `.git`.
La versión `v-docker` no se versiona en Git (ver `.gitignore`).

## 12. Fuentes

- Conjunto de datos *Telco Customer Churn* (IBM):
  https://github.com/IBM/telco-customer-churn-on-icp4d

## 13. Control de versiones (Git)

Flujo: `main` (estable), `develop` (integración), `feature/*` (tareas). La
Entrega 2 se integra en `main` y se identifica con la etiqueta `entrega-2`.