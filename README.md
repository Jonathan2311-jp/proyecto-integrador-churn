# Proyecto Integrador — Predicción de fuga de clientes (churn)

**Curso:** Modelos y Simulación de Sistemas I — 2026-II · 

**Docente:** Andrés Parra

**Integrantes:** Jonathan Alfonso Pérez · Zareth Mariana Vega Sánchez  

Modelo de clasificación binaria que estima si un cliente de telecomunicaciones
cancelará su servicio (`Churn`). El proyecto es acumulativo: parte de un notebook
y avanza hacia una aplicación reproducible, contenerizada y, en las fases
siguientes, accesible por API REST y con monitoreo.

**Datos:** *Telco Customer Churn* (IBM) — 7.043 clientes, 19 variables
predictoras y ~26,5 % de cancelaciones.
Fuente: https://github.com/IBM/telco-customer-churn-on-icp4d

## Resumen por fase

### Fase 1 — Modelo predictivo (completa) · [`fase-1/`](fase-1/README.md)

Análisis exploratorio, tratamiento de los 11 valores faltantes de `TotalCharges`
(clientes nuevos con `tenure = 0`, imputados con 0), partición estratificada
80/20 sin fuga de información y comparación de regresión logística contra
Random Forest. El modelo final es una regresión logística (`C = 0.1`) guardada
junto con su preprocesamiento. En prueba logra recall 0.7834, F1 0.6181 y
ROC-AUC 0.8413, frente a 0 de recall y 0.5 de ROC-AUC del modelo base.

### Fase 2 — Scripts y Docker (completa) · [`fase-2/`](fase-2/README.md)

El notebook se convierte en una aplicación por línea de comandos: paquete
`src/` reutilizable, `train.py` y `predict.py` con `argparse`, validación de
columnas, tipos y nulos, y versiones del modelo en `models/v1` y `models/v2`
con sus métricas y metadatos (un identificador repetido no se sobrescribe).
Incluye `Dockerfile` (entrenamiento y predicción en contenedor, con datos y
resultados montados desde el anfitrión), 12 pruebas con `pytest` y evidencia de
ejecución en el README.

### Fase 3 — API REST (pendiente)

Servicios `/health`, `/predict` y `/train` sobre el código de la Fase 2.

### Fase 4 — Monitoreo básico (pendiente)

Principios básicos de monitoreo y gestión del modelo desplegado.

## Estructura del repositorio

| Carpeta | Contenido |
|---|---|
| [`fase-1/`](fase-1/README.md) | `notebook.ipynb`, datos y `modelo.joblib` |
| [`fase-2/`](fase-2/README.md) | `train.py`, `predict.py`, `src/`, `tests/`, `Dockerfile`, `models/` |

## Cómo empezar

Las instrucciones de entorno, entrenamiento, predicción, Docker y pruebas
están en [`fase-2/README.md`](fase-2/README.md).

## Flujo de trabajo (Git)

`main` (versiones entregadas) · `develop` (integración) · `feature/*` (tareas,
integradas mediante Pull Request revisado por otro integrante). La Entrega 2 se
identifica con la etiqueta `entrega-2`.
