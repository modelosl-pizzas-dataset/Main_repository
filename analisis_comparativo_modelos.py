# =====================================================
# COMPARACIÓN DE 3 DATASETS DE PIZZA SALES
# Objetivo: predecir 'unit_price'
# =====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Modelos
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# =====================================================
# 1. CONFIGURACIÓN
# =====================================================

DATASETS = [
    {"nombre": "Pizza Sin Tiempo",       "ruta": "data/pizza_sales_sin_tiempo.csv",       "objetivo": "unit_price"},
    {"nombre": "Pizza Pearson-Spearman", "ruta": "data/pizza_sales_pearson_spearman.csv", "objetivo": "unit_price"},
]

TEST_SIZE = 0.2
RANDOM_STATE = 42

# =====================================================
# 2. DEFINICIÓN DE MODELOS
# =====================================================

def get_modelos():
    return {
        "Regresión Lineal": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Lasso": Lasso(alpha=0.1, random_state=RANDOM_STATE),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE),
        "Árbol de Decisión": DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_STATE),
        "SVM (RBF)": SVR(kernel='rbf', C=1.0, gamma='scale'),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "Red Neuronal (MLP)": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=RANDOM_STATE),
    }

# =====================================================
# 3. FUNCIONES AUXILIARES
# =====================================================

def cargar_y_preparar(ruta, objetivo):
    df = pd.read_csv(ruta)
    
    # Separar X e y
    X = df.drop(columns=[objetivo])
    y = df[objetivo]
    
    # Codificar variables categóricas
    columnas_categoricas = X.select_dtypes(include=['object']).columns.tolist()
    if columnas_categoricas:
        X = pd.get_dummies(X, columns=columnas_categoricas, drop_first=True)
    
    # Imputar nulos con la media
    X = X.fillna(X.mean())
    
    return X, y

def evaluar_modelo(modelo, X_train, X_test, y_train, y_test):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('modelo', modelo)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {"RMSE": rmse, "MAE": mae, "R2": r2}, pipeline

# =====================================================
# 4. BUCLE PRINCIPAL
# =====================================================

resultados = []

for ds in DATASETS:
    print("\n" + "="*70)
    print(f"EVALUANDO: {ds['nombre']}")
    print("="*70)
    
    try:
        X, y = cargar_y_preparar(ds["ruta"], ds["objetivo"])
        print(f"Dataset cargado: {X.shape[0]} filas, {X.shape[1]} columnas")
    except Exception as e:
        print(f"❌ Error al cargar {ds['nombre']}: {e}")
        continue
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    modelos = get_modelos()
    for nombre_modelo, modelo in modelos.items():
        try:
            metricas, _ = evaluar_modelo(modelo, X_train, X_test, y_train, y_test)
            resultados.append({
                "Dataset": ds["nombre"],
                "Modelo": nombre_modelo,
                "RMSE": round(metricas["RMSE"], 4),
                "MAE": round(metricas["MAE"], 4),
                "R2": round(metricas["R2"], 4),
            })
            print(f"  ✅ {nombre_modelo:25s} → RMSE: {metricas['RMSE']:.4f}, R²: {metricas['R2']:.4f}")
        except Exception as e:
            print(f"  ❌ {nombre_modelo:25s} → Error: {e}")

# =====================================================
# 5. RESULTADOS
# =====================================================

df_resultados = pd.DataFrame(resultados)
print("\n" + "="*70)
print("RESULTADOS COMPLETOS")
print("="*70)
print(df_resultados.to_string(index=False))

df_resultados.to_csv("resultados_comparacion.csv", index=False)
print("\n✅ Resultados guardados en 'resultados_comparacion.csv'")

# =====================================================
# 6. VISUALIZACIÓN
# =====================================================

plt.figure(figsize=(14, 6))
sns.barplot(data=df_resultados, x="Modelo", y="RMSE", hue="Dataset")
plt.title("Comparación de RMSE por Modelo y Dataset")
plt.xticks(rotation=45, ha='right')
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
sns.barplot(data=df_resultados, x="Modelo", y="R2", hue="Dataset")
plt.title("Comparación de R² por Modelo y Dataset")
plt.xticks(rotation=45, ha='right')
plt.grid()
plt.tight_layout()
plt.show()

# =====================================================
# 7. RANKING
# =====================================================

print("\n" + "="*70)
print("RANKING DE RESULTADOS")
print("="*70)

print("\n📊 MEJOR MODELO POR DATASET (menor RMSE):")
for ds_nombre in df_resultados["Dataset"].unique():
    subset = df_resultados[df_resultados["Dataset"] == ds_nombre]
    mejor = subset.loc[subset["RMSE"].idxmin()]
    print(f"  {ds_nombre}: {mejor['Modelo']} (RMSE: {mejor['RMSE']}, R²: {mejor['R2']})")

print("\n📊 MEJOR DATASET (menor RMSE promedio):")
promedio_por_dataset = df_resultados.groupby("Dataset")["RMSE"].mean().sort_values()
print(promedio_por_dataset)

print("\n📊 MEJOR MODELO GLOBAL (menor RMSE promedio):")
promedio_por_modelo = df_resultados.groupby("Modelo")["RMSE"].mean().sort_values()
print(promedio_por_modelo)

print("\n📊 RANKING FINAL (ordenado por RMSE):")
ranking = df_resultados.sort_values("RMSE").reset_index(drop=True)
ranking.index = ranking.index + 1
print(ranking.to_string())

# =====================================================
# 8. CONCLUSIÓN
# =====================================================

print("\n" + "="*70)
print("CONCLUSIÓN")
print("="*70)

mejor_dataset = promedio_por_dataset.index[0]
mejor_modelo = promedio_por_modelo.index[0]

print(f"\n🏆 Mejor dataset: {mejor_dataset}")
print(f"🏆 Mejor modelo: {mejor_modelo}")

mejor_combinacion = df_resultados.loc[df_resultados["RMSE"].idxmin()]
print(f"\n🏆 Mejor combinación: {mejor_combinacion['Modelo']} en {mejor_combinacion['Dataset']}")
print(f"   RMSE: {mejor_combinacion['RMSE']}")
print(f"   MAE: {mejor_combinacion['MAE']}")
print(f"   R²: {mejor_combinacion['R2']}")

