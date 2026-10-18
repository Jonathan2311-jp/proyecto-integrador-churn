# Proyecto Integrador — Fase 1: Predicción de fuga de clientes (churn)

**Curso:** Modelos y Simulación de Sistemas I — 2026-II
**Integrantes:** Zareth Mariana Vega Sánchez · Jonathan Alfonso Pérez

## Problema

Una empresa de telecomunicaciones necesita anticipar qué clientes cancelarán
el servicio, para enfocar las acciones de retención antes de que la
cancelación ocurra. Es un problema de **aprendizaje supervisado, clasificación
binaria**: la variable objetivo `Churn` indica si el cliente canceló (`Yes`) o
permanece (`No`). Las clases están desbalanceadas (~26,5 % cancelan).

## Datos

- **Fuente:** conjunto *Telco Customer Churn* (IBM).
- **Archivo:** `data/telco_churn.csv` — 7.043 clientes, 19 variables predictoras
  (demográficas, servicios contratados y facturación) + el identificador
  `customerID` + la variable objetivo `Churn`.

## Qué se hizo

1. **Exploración de los datos:** tipos de variables, valores faltantes,
   distribuciones, relación de cada variable con la cancelación y correlaciones.
2. **Preparación:** se eliminó `customerID`, se convirtió `TotalCharges` a
   numérico y se imputaron sus 11 faltantes con 0 (corresponden a clientes
   nuevos con `tenure = 0`, que aún no tienen factura acumulada).
3. **Partición estratificada** 80/20 (`random_state = 42`), conservando la
   proporción de clases y evitando fuga de información.
4. **Modelado:** se compararon **regresión logística** y **Random Forest**
   (ambos con `class_weight="balanced"`), ajustando hiperparámetros por
   validación cruzada estratificada de 5 particiones con F1 como criterio.
5. **Modelo final:** regresión logística con `C = 0.1`, encapsulada junto a su
   preprocesamiento en un `Pipeline` y guardada en `modelo.joblib`. Se eligió
   sobre el Random Forest por desempeño equivalente y mayor interpretabilidad.

## Resultados (conjunto de prueba)

| Métrica | Modelo base | Regresión logística |
|---|---|---|
| Accuracy | 0.7346 | 0.7431 |
| Recall | 0.0000 | 0.7834 |
| F1 | 0.0000 | 0.6181 |
| ROC-AUC | 0.5000 | 0.8413 |

El modelo detecta el 78 % de las cancelaciones reales, frente a ninguna del
modelo base (que predice siempre la clase mayoritaria).

## Archivos

- `notebook.ipynb` — notebook completo (exploración, preparación, modelado,
  evaluación y guardado).
- `data/telco_churn.csv` — conjunto de datos.
- `modelo.joblib` — pipeline entrenado (preprocesamiento + modelo).

## Cómo ejecutar

Abrir `notebook.ipynb` en Jupyter o VS Code y ejecutar todas las celdas en
orden. Requiere `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn` y
`joblib`.

## Fuentes

- Conjunto *Telco Customer Churn* (IBM):
  https://github.com/IBM/telco-customer-churn-on-icp4d