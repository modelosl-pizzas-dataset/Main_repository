from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import root_mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = Path(__file__).resolve().parent
ARCHIVO_CSV = BASE_DIR / "data" / "pizza_sales.csv"

df = pd.read_csv(ARCHIVO_CSV)

df["n_ingredientes"] = df["pizza_ingredients"].str.split(",").apply(len)
df["tiene_pollo"] = df["pizza_ingredients"].str.contains("chicken", case=False)
df["tiene_carnes"] = df["pizza_ingredients"].str.contains(
    "pepperoni|bacon|sausage|salami", case=False
)
df["tiene_vegetales"] = df["pizza_ingredients"].str.contains(
    "pepper|onion|mushroom|spinach|tomato", case=False
)

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

for col in ["pizza_size", "pizza_category"]:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop(columns=["unit_price"])
y = df["unit_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

modelos = {
    "Regresión Lineal": LinearRegression(),
    "Árbol de Decisión": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        random_state=42,
        n_estimators=100
    )
}

resultados = []

for nombre, modelo in modelos.items():

    modelo.fit(X_train, y_train)

    pred = modelo.predict(X_test)

    rmse = root_mean_squared_error(y_test, pred)

    resultados.append([nombre, rmse])

resultados = pd.DataFrame(resultados, columns=["Modelo", "RMSE"])

print("\nComparación de modelos")
print(resultados.sort_values("RMSE"))

mejor = resultados.sort_values("RMSE").iloc[0]

print(f"\nMejor modelo: {mejor['Modelo']}")
print(f"RMSE: {mejor['RMSE']:.4f}")