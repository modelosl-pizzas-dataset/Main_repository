"""Constantes compartidas por entrenamiento y predicción."""

TARGET = "unit_price"
GROUP_COLUMN = "pizza_name"          # grupo natural: tipo de pizza (solo para separar, NO es variable de entrada)
RAW_COLUMNS = ["pizza_size", "pizza_category", "pizza_ingredients"]

# Orden natural de los tamaños (ordinal). Un tamaño fuera de esta lista es un error de entrada.
SIZE_ORDER = ["S", "M", "L", "XL", "XXL"]

# Esquema de entrada: columna -> tipo esperado (todas son texto)
INPUT_SCHEMA = {
    "pizza_size": "texto",
    "pizza_category": "texto",
    "pizza_ingredients": "texto",
}

# Columnas que admiten nulos (se imputan con la regla aprendida en entrenamiento)
NULLABLE_COLUMNS = ["pizza_size", "pizza_category", "pizza_ingredients"]

DEFAULT_RF_PARAMS = {"n_estimators": 200, "max_depth": None}
