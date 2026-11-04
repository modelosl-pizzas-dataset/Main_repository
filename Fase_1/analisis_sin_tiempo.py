from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import root_mean_squared_error

# Ruta al dataset
BASE_DIR = Path(__file__).resolve().parent
ARCHIVO_CSV = BASE_DIR / "data" / "pizza_sales.csv"
ARCHIVO_COPIA = BASE_DIR / "data" / "pizza_sales_sin_tiempo.csv"

# Cargar datos
df = pd.read_csv(ARCHIVO_CSV)

# =========================
# Ingeniería de variables
# =========================

df["n_ingredientes"] = df["pizza_ingredients"].str.split(",").apply(len)

ingredientes = (
    df["pizza_ingredients"]
    .fillna("")
    .str.split(",")
    .explode()
    .str.strip()
)
ingredientes = ingredientes[ingredientes.ne("")]
ingredientes_one_hot = pd.get_dummies(
    ingredientes,
    prefix="ingrediente",
    dtype=int
).groupby(level=0).max()
df = df.join(ingredientes_one_hot)

# =========================
# Eliminar variables de tiempo
# =========================

df = df.drop(columns=[
    "pizza_id",
    "order_id",
    "pizza_name",
    "pizza_name_id",
    "pizza_ingredients",
    "total_price",
    "order_date",
    "order_time"
])

# =========================
# Codificar variables categóricas
# =========================

for col in ["pizza_size", "pizza_category"]:
    df[col] = LabelEncoder().fit_transform(df[col])

if ARCHIVO_COPIA.exists():
    print(f"La copia ya existe y está alojada en: {ARCHIVO_COPIA}")
else:
    print(f"No existe la copia; se creará en: {ARCHIVO_COPIA}")

df.to_csv(ARCHIVO_COPIA, index=False)
df = pd.read_csv(ARCHIVO_COPIA)

# =========================
# Preparar entrenamiento
# =========================

X = df.drop(columns=["unit_price"])
y = df["unit_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# Entrenar modelo
# =========================

modelo = RandomForestRegressor(
    random_state=42,
    n_estimators=100
)

modelo.fit(X_train, y_train)

pred = modelo.predict(X_test)

rmse = root_mean_squared_error(y_test, pred)

# =========================
# Resultados
# =========================

print("\nANÁLISIS SIN VARIABLES DE TIEMPO")
print("-" * 40)
print(f"RMSE: {rmse:.4f}")

# Importancia de variables
importancias = pd.DataFrame({
    "Variable": X.columns,
    "Importancia": modelo.feature_importances_
}).sort_values("Importancia", ascending=False)

print("\nImportancia de variables:")
print(importancias)