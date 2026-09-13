# =====================================================
# SCRIPT DE ANÁLISIS DE DATASET Y SELECCIÓN DE VARIABLES
# =====================================================

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# =====================================================
# 1. CONFIGURACIÓN (¡CAMBIA ESTOS VALORES!)
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
ARCHIVO_CSV = BASE_DIR / "data" / "pizza_sales.csv"
COLUMNA_OBJETIVO = "unit_price"
TIPO_PROBLEMA = "regresion"  # "clasificacion" o "regresion"
UMBRAL_NULOS = 0.5               # Filas con >50% de nulos
UMBRAL_IMPORTANCIA = 0.001       # Importancia mínima para considerar una variable útil

# =====================================================
# 2. CARGAR DATOS
# =====================================================

df = pd.read_csv(ARCHIVO_CSV)
print(f"✅ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"\nPrimeras 5 filas:")
print(df.head())

# =====================================================
# 3. ANÁLISIS EXPLORATORIO BÁSICO
# =====================================================

print("\n" + "="*60)
print("ANÁLISIS EXPLORATORIO")
print("="*60)

# 3.1. Tipos de datos
print("\n📊 Tipos de datos:")
print(df.dtypes.value_counts())

# 3.2. Valores nulos
print("\n📊 Valores nulos por columna:")
nulos = df.isnull().sum()
nulos = nulos[nulos > 0].sort_values(ascending=False)
if len(nulos) > 0:
    print(nulos)
else:
    print("No hay valores nulos.")

# 3.3. Duplicados
print(f"\n📊 Filas duplicadas: {df.duplicated().sum()}")

# 3.4. Columnas con un solo valor único
print("\n📊 Columnas con un solo valor único (candidatas a eliminar):")
columnas_constantes = [col for col in df.columns if df[col].nunique() == 1]
if columnas_constantes:
    for col in columnas_constantes:
        print(f"  - {col}: {df[col].unique()}")
else:
    print("No hay columnas constantes.")

# 3.5. Columnas con muchos valores únicos (posibles IDs)
print("\n📊 Columnas con muchos valores únicos (posibles IDs):")
for col in df.columns:
    if df[col].nunique() > 0.9 * len(df):
        print(f"  - {col}: {df[col].nunique()} valores únicos")

# =====================================================
# 4. DISTRIBUCIÓN DE LA VARIABLE OBJETIVO
# =====================================================

print("\n" + "="*60)
print("DISTRIBUCIÓN DE LA VARIABLE OBJETIVO")
print("="*60)

if TIPO_PROBLEMA == "clasificacion":
    print(f"\n📊 Distribución de '{COLUMNA_OBJETIVO}':")
    print(df[COLUMNA_OBJETIVO].value_counts())
    print(f"\nProporción:")
    print(df[COLUMNA_OBJETIVO].value_counts(normalize=True).round(3))
    
    plt.figure(figsize=(8, 4))
    df[COLUMNA_OBJETIVO].value_counts().plot(kind='bar')
    plt.title(f"Distribución de {COLUMNA_OBJETIVO}")
    plt.xlabel(COLUMNA_OBJETIVO)
    plt.ylabel("Frecuencia")
    plt.grid()
    plt.show()

else:
    print(f"\n📊 Estadísticas de '{COLUMNA_OBJETIVO}':")
    print(df[COLUMNA_OBJETIVO].describe())
    
    plt.figure(figsize=(8, 4))
    df[COLUMNA_OBJETIVO].hist(bins=30)
    plt.title(f"Distribución de {COLUMNA_OBJETIVO}")
    plt.xlabel(COLUMNA_OBJETIVO)
    plt.ylabel("Frecuencia")
    plt.grid()
    plt.show()

# =====================================================
# 5. CORRELACIONES
# =====================================================

print("\n" + "="*60)
print("CORRELACIONES CON LA VARIABLE OBJETIVO")
print("="*60)

numericas = df.select_dtypes(include=[np.number]).columns.tolist()

if COLUMNA_OBJETIVO in numericas:
    correlaciones = df[numericas].corr()[COLUMNA_OBJETIVO].sort_values(ascending=False)
    print(f"\n📊 Correlación de cada variable con '{COLUMNA_OBJETIVO}':")
    print(correlaciones)
    
    plt.figure(figsize=(10, 6))
    correlaciones.drop(COLUMNA_OBJETIVO).plot(kind='barh')
    plt.title(f"Correlación con {COLUMNA_OBJETIVO}")
    plt.xlabel("Correlación")
    plt.grid()
    plt.show()
else:
    print(f"'{COLUMNA_OBJETIVO}' no es numérica. No se puede calcular correlación.")

# =====================================================
# 6. IMPORTANCIA DE VARIABLES CON RANDOM FOREST
# =====================================================

print("\n" + "="*60)
print("IMPORTANCIA DE VARIABLES CON RANDOM FOREST")
print("="*60)

df_model = df.copy()
columnas_categoricas = df_model.select_dtypes(include=['object', 'string', 'str']).columns.tolist()
if columnas_categoricas:
    print(f"\n📊 Codificando {len(columnas_categoricas)} columnas categóricas...")
    df_model = pd.get_dummies(df_model, columns=columnas_categoricas, drop_first=True)

X = df_model.drop(columns=[COLUMNA_OBJETIVO])
y = df_model[COLUMNA_OBJETIVO]
X = X.fillna(0)
y = y.fillna(0) if TIPO_PROBLEMA == "regresion" else y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

if TIPO_PROBLEMA == "clasificacion":
    modelo = RandomForestClassifier(
        n_estimators=20,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
else:
    modelo = RandomForestRegressor(
        n_estimators=20,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
if TIPO_PROBLEMA == "clasificacion":
    print(f"\n📊 Accuracy en test: {accuracy_score(y_test, y_pred):.3f}")
else:
    print(f"\n📊 RMSE en test: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")

importancias = pd.DataFrame({
    'Variable': X.columns,
    'Importancia': modelo.feature_importances_
}).sort_values('Importancia', ascending=False)

print(f"\n📊 Top 20 variables más importantes:")
print(importancias.head(20).to_string(index=False))

plt.figure(figsize=(10, 8))
plt.barh(importancias['Variable'].head(20)[::-1], importancias['Importancia'].head(20)[::-1])
plt.xlabel("Importancia")
plt.title("Top 20 Variables Más Importantes")
plt.grid()
plt.show()

# =====================================================
# 7. IDENTIFICAR VARIABLES INÚTILES Y FILAS A ELIMINAR
# =====================================================

print("\n" + "="*60)
print("VARIABLES INÚTILES Y FILAS A ELIMINAR")
print("="*60)

# --- 7.1. Variables inútiles ---

variables_inutiles = []

# Columnas constantes
if columnas_constantes:
    variables_inutiles.extend(columnas_constantes)

# Posibles IDs
ids = []
for col in df.columns:
    if col != COLUMNA_OBJETIVO and df[col].nunique() > 0.9 * len(df):
        ids.append(col)
if ids:
    variables_inutiles.extend(ids)

# Variables de baja importancia
baja_importancia = importancias[importancias['Importancia'] <= UMBRAL_IMPORTANCIA]['Variable'].tolist()
if baja_importancia:
    variables_inutiles.extend(baja_importancia)

# --- 7.2. Filas a eliminar ---

# Duplicados
duplicados = df.duplicated().sum()

# Filas con demasiados nulos
nulos_por_fila = df.isnull().sum(axis=1)
filas_muchos_nulos = nulos_por_fila[nulos_por_fila > UMBRAL_NULOS * df.shape[1]]

# --- 7.3. Imprimir resumen ---

print("\n" + "="*60)
print("RESUMEN FINAL")
print("="*60)

print(f"\n📊 VARIABLES INÚTILES ({len(variables_inutiles)}):")
for var in variables_inutiles:
    print(f"  - {var}")

print(f"\n📊 FILAS A ELIMINAR:")
print(f"  - Duplicados: {duplicados}")
print(f"  - Con muchos nulos: {len(filas_muchos_nulos)}")
if len(filas_muchos_nulos) > 0:
    print(f"    Índices: {filas_muchos_nulos.index.tolist()[:20]}...")