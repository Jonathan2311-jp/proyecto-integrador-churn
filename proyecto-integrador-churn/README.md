# proyecto-integrador-churn
Fase 1 – Prediccion de fuga de clientes (churn). Modelos y Simulacion de Sistemas I 2026-II

Integrantes del equipo

Zareth Mariana Vega Sanchez
Jonathan Alfonso Perez

Descripcion del problema

Una empresa de telecomunicaciones necesita anticipar que clientes cancelaran su servicio (churn). Actualmente la cancelacion solo se conoce cuando ya ocurrio, lo que impide cualquier accion de retencion. El proyecto aborda este problema como una tarea de clasificacion binaria supervisada: estimar, para cada cliente, la probabilidad de que cancele el servicio a partir de su perfil demografico, los servicios contratados y sus condiciones de contrato y facturacion.

Fuente del conjunto de datos

Telco Customer Churn, publicado originalmente por IBM como conjunto de datos de muestra y ampliamente distribuido a traves de Kaggle (enlace). El notebook carga los datos directamente desde el repositorio publico de IBM en GitHub, para que la ejecucion no dependa de subir un CSV al repositorio:

https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

Este contiene 7.043 observaciones/clientes y 21 columnas (19 predictoras, 1 identificador y la variable objetivo Churn).

Objetivo del modelo

Predecir si un cliente cancelara el servicio (Churn = Yes) o permanecera activo (Churn = No), priorizando la deteccion de clientes en riesgo para enfocar campañas de retencion, en lugar de aplicarlas de forma indiscriminada a toda la base de clientes.