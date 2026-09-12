import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#%% md
## 1. Carga de datos y split — NO MODIFICARcccccccccccccccccccccccccccc
#%%cccccccccccccccccccc
# ⚠️ No cambies nada en esta celda.
# El mismo dataset y el mismo split deben usarlo TODOS los participantes.

data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Filas de entrenamiento:", X_train.shape[0])
print("Filas de prueba:", X_test.shape[0])
X_train.head()

#%%
# TODO: cambia este valor por el modelo que te fue asignado
# Ejemplos válidos: "regresion-lineal", "ridge", "arbol-decision", "random-forest", "knn"
NOMBRE_MODELO = "cambia-esto"


#%%
# --- Ejemplos (descomenta el que te corresponda) ---

# Regresión lineal
# from sklearn.linear_model import LinearRegression
# modelo = LinearRegression()

# Ridge
# from sklearn.linear_model import Ridge
# modelo = Ridge(alpha=1.0, random_state=42)

# Árbol de decisión
# from sklearn.tree import DecisionTreeRegressor
# modelo = DecisionTreeRegressor(max_depth=6, random_state=42)

# Random Forest
# from sklearn.ensemble import RandomForestRegressor
# modelo = RandomForestRegressor(n_estimators=200, random_state=42)

# KNN
# from sklearn.neighbors import KNeighborsRegressor
# modelo = KNeighborsRegressor(n_neighbors=5)

# TODO: descomenta tu modelo arriba, o escribe el tuyo aquí
modelo = None

modelo.fit(X_train, y_train)
#%% md
## 4. Calcula tus métricas — NO MODIFICAR
#%%
y_pred = modelo.predict(X_test)

# np.sqrt() funciona en cualquier versión de scikit-learn
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")

#%%
ruta_csv = "../resultados/metricas.csv"

df = pd.read_csv(ruta_csv)

nueva_fila = pd.DataFrame([{
    "modelo": NOMBRE_MODELO,
    "rmse": round(rmse, 4),
    "mae": round(mae, 4),
    "r2": round(r2, 4),
}])

df = df[df["modelo"] != NOMBRE_MODELO]  # evita duplicados si vuelves a correr el notebook
df = pd.concat([df, nueva_fila], ignore_index=True)
df.to_csv(ruta_csv, index=False)

df

