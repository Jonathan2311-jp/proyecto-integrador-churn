"""Construccion del preprocesador (ColumnTransformer) del modelo de churn."""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .datos import COLS_NUM, COLS_BIN, COLS_CAT


def construir_preprocesador(escalar=True):
    """ColumnTransformer: numericas (escaladas o no), binaria passthrough,
    categoricas con One-Hot (drop='first', ignora categorias no vistas)."""
    transformador_num = StandardScaler() if escalar else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", transformador_num, COLS_NUM),
            ("bin", "passthrough", COLS_BIN),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), COLS_CAT),
        ],
        remainder="drop",
    )