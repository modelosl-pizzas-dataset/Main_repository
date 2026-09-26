# =====================================================
# MAIN - MODELO FINAL (RANDOM FOREST)
# Dataset óptimo: pizza_sales_sin_tiempo.csv
# Objetivo: predecir 'unit_price'
# =====================================================

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =====================================================
# 1. CONFIGURACIÓN
# =====================================================

RUTA_DATASET = "data/pizza_sales_sin_tiempo.csv"
COLUMNA_OBJETIVO = "unit_price"
RANDOM_STATE = 42
TEST_SIZE = 0.2

print("="*60)
print("MAIN - MODELO FINAL (RANDOM FOREST)")
print("="*60)
print(f"Dataset: {RUTA_DATASET}")
print(f"Objetivo: {COLUMNA_OBJETIVO}")
print(f"Modelo: Random Forest")
print("="*60)

# =====================================================
# 2. CARGAR DATOS
# =====================================================

df = pd.read_csv(RUTA_DATASET)
print(f"\n✅ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

# Separar X e y
X = df.drop(columns=[COLUMNA_OBJETIVO])
y = df[COLUMNA_OBJETIVO]

# Codificar variables categóricas (si las hay)
columnas_categoricas = X.select_dtypes(include=['object']).columns.tolist()
if columnas_categoricas:
    print(f"📊 Codificando {len(columnas_categoricas)} columnas categóricas...")
    X = pd.get_dummies(X, columns=columnas_categoricas, drop_first=True)

# Imputar valores nulos con la media
X = X.fillna(X.mean())

print(f"📊 Features: {X.shape[1]} columnas")
print(f"📊 Target: {y.shape[0]} valores")

# =====================================================
# 3. SPLIT TRAIN/TEST
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print(f"\n📊 Train: {X_train.shape[0]} filas")
print(f"📊 Test:  {X_test.shape[0]} filas")

# =====================================================
# 4. ENTRENAR MODELO: RANDOM FOREST
# =====================================================

print("\n" + "="*60)
print("ENTRENANDO RANDOM FOREST")
print("="*60)

modelo = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

modelo.fit(X_train, y_train)
print("✅ Modelo entrenado")

# =====================================================
# 5. EVALUAR MODELO
# =====================================================

y_pred = modelo.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*60)
print("MÉTRICAS DEL MODELO")
print("="*60)
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")

# =====================================================
# 6. GUARDAR MODELO
# =====================================================

joblib.dump(modelo, "modelo_entrenado.pkl")
print("\n✅ Modelo guardado como 'modelo_entrenado.pkl'")

# =====================================================
# 7. GUARDAR DATASET OPTIMIZADO
# =====================================================

# Guardar el dataset con las features ya codificadas
dataset_optimizado = X.copy()
dataset_optimizado[COLUMNA_OBJETIVO] = y

dataset_optimizado.to_csv("dataset_optimizado.csv", index=False)
print("✅ Dataset optimizado guardado como 'dataset_optimizado.csv'")
print(f"   Shape: {dataset_optimizado.shape}")

# =====================================================
# 8. GUARDAR MÉTRICAS
# =====================================================

metricas = pd.DataFrame([{
    "dataset": RUTA_DATASET,
    "modelo": "Random Forest",
    "rmse": round(rmse, 4),
    "mae": round(mae, 4),
    "r2": round(r2, 4),
}])
metricas.to_csv("metricas_finales.csv", index=False)
print("✅ Métricas guardadas como 'metricas_finales.csv'")

# =====================================================
# 9. IMPORTANCIA DE VARIABLES
# =====================================================

importancias = pd.DataFrame({
    "Variable": X.columns,
    "Importancia": modelo.feature_importances_
}).sort_values("Importancia", ascending=False)

print("\n" + "="*60)
print("TOP 15 VARIABLES MÁS IMPORTANTES")
print("="*60)
print(importancias.head(15).to_string(index=False))

# Guardar importancias
importancias.to_csv("importancia_variables.csv", index=False)
print("\n✅ Importancia de variables guardada como 'importancia_variables.csv'")

# Visualización
plt.figure(figsize=(10, 6))
plt.barh(importancias["Variable"].head(15)[::-1], importancias["Importancia"].head(15)[::-1])
plt.xlabel("Importancia")
plt.title("Top 15 Variables Más Importantes (Random Forest)")
plt.grid()
plt.tight_layout()
plt.savefig("importancia_variables.png", dpi=100, bbox_inches='tight')
plt.show()
print("✅ Gráfico guardado como 'importancia_variables.png'")

# =====================================================
# 10. VERIFICACIÓN FINAL
# =====================================================

print("\n" + "="*60)
print("RESUMEN DE ARCHIVOS GENERADOS")
print("="*60)
print("✅ modelo_entrenado.pkl         → Modelo Random Forest")
print("✅ dataset_optimizado.csv       → Dataset con features codificadas")
print("✅ metricas_finales.csv         → RMSE, MAE, R²")
print("✅ importancia_variables.csv    → Importancia de cada variable")
print("✅ importancia_variables.png    → Gráfico de importancia")
print("\n🎯 Listo para la siguiente fase (train/test).")