# Predicción del precio unitario de pizzas (Pizza Sales)

Proyecto de análisis y modelado predictivo sobre el dataset **Pizza Sales** (48,620 pedidos, 2015). El objetivo es predecir la variable `unit_price` (precio unitario de cada pizza) a partir de sus características —tamaño, categoría e ingredientes—, comparando distintos algoritmos de regresión y seleccionando el mejor modelo final.

## Estructura del repositorio

```
Main_repository-main/
├── data/
│   ├── pizza_sales.csv                     # Dataset original (crudo)
│   ├── pizza_sales_sin_tiempo.csv          # Generado por analisis_sin_tiempo.py
│   └── pizza_sales_pearson_spearman.csv    # Generado por analisis_pearson_spearman.py
├── analisis.py                             # EDA general + selección de variables (no guarda nada)
├── analisis_sin_tiempo.py                  # Genera pizza_sales_sin_tiempo.csv
├── analisis_pearson_spearman.py            # Genera pizza_sales_pearson_spearman.csv
├── analisis_comparativo_modelos.py         # Compara 10 modelos → resultados_comparacion.csv
├── main.py                                 # Entrena el modelo final (Random Forest)
├── modelo_entrenado.pkl                    # Salida de main.py
├── dataset_optimizado.csv                  # Salida de main.py
├── metricas_finales.csv                    # Salida de main.py
├── resultados_comparacion.csv              # Salida de analisis_comparativo_modelos.py
├── importancia_variables.csv               # Salida de main.py
└── importancia_variables.png               # Salida de main.py
```

## ⚠️ Importante: ¿qué archivos se crean y cuándo?

Este punto genera confusión porque **no todos los scripts se comportan igual**, así que quedó documentado explícitamente:

| Script | ¿Qué necesita que ya exista? | ¿Qué genera/sobreescribe cada vez que corre? |
|---|---|---|
| `analisis.py` | `data/pizza_sales_sin_tiempo.csv` | **Nada.** Solo imprime en consola y muestra gráficos con `plt.show()`. No guarda ningún archivo. |
| `analisis_sin_tiempo.py` | `data/pizza_sales.csv` (el crudo) | `data/pizza_sales_sin_tiempo.csv` — **se sobreescribe siempre**, sin importar si ya existía. |
| `analisis_pearson_spearman.py` | `data/pizza_sales.csv` (el crudo) | `data/pizza_sales_pearson_spearman.csv` — **se sobreescribe siempre**, sin importar si ya existía. |
| `analisis_comparativo_modelos.py` | `data/pizza_sales_sin_tiempo.csv` y `data/pizza_sales_pearson_spearman.csv` (ya deben existir) | `resultados_comparacion.csv` — se sobreescribe cada vez, pero solo con los datasets que sí pudo cargar (ver nota abajo). |
| `main.py` | `data/pizza_sales_sin_tiempo.csv` (ya debe existir) | `modelo_entrenado.pkl`, `dataset_optimizado.csv`, `metricas_finales.csv`, `importancia_variables.csv`, `importancia_variables.png` — **los 5 se sobreescriben cada vez que corres el script**. |

**Puntos clave:**

- Los dos scripts que "preparan" los datos (`analisis_sin_tiempo.py` y `analisis_pearson_spearman.py`) imprimen un mensaje como *"La copia ya existe y está alojada en..."* si el CSV de salida ya está en `data/`, **pero ese mensaje es solo informativo**: el código llama a `df.to_csv(...)` inmediatamente después sin ningún `if` que lo evite, así que el archivo se reescribe igual, exista o no.
- **`main.py` NO genera los datasets procesados de `data/`.** Simplemente hace `pd.read_csv("data/pizza_sales_sin_tiempo.csv")` asumiendo que ya existe. Si lo borras y corres `main.py` directamente, el script **falla con `FileNotFoundError` y se detiene por completo** (no tiene manejo de errores). Por eso el orden de ejecución (ver abajo) importa: primero hay que generar los datasets con `analisis_sin_tiempo.py` (y opcionalmente `analisis_pearson_spearman.py`), y solo después correr `main.py`.
- `analisis_comparativo_modelos.py` sí tiene manejo de errores por dataset: si a alguno de los dos (`pizza_sales_sin_tiempo.csv` o `pizza_sales_pearson_spearman.csv`) no lo encuentra, captura la excepción, imprime un mensaje de error y **continúa con el otro** en vez de detener todo el script. Esto es distinto al comportamiento de `main.py`, que sí truena si falta su archivo.
- Los datasets `pizza_sales_sin_tiempo.csv` y `pizza_sales_pearson_spearman.csv` que vienen incluidos en este repo (dentro de `data/`) **ya fueron generados en una ejecución previa** de esos scripts; no hace falta volver a correrlos si no vas a cambiar la ingeniería de variables. Solo se regeneran si los borras o si corres los scripts de nuevo.
- `analisis.py` es puramente exploratorio: puedes correrlo las veces que quieras, nunca modifica ni crea archivos en disco.

## Dataset

- **Fuente:** `data/pizza_sales.csv` — 48,620 filas, 12 columnas (pedidos de pizza de un año completo).
- **Columnas originales:** `pizza_id`, `order_id`, `pizza_name_id`, `quantity`, `order_date`, `order_time`, `unit_price`, `total_price`, `pizza_size` (S/M/L/XL/XXL), `pizza_category` (Classic/Veggie/Supreme/Chicken), `pizza_ingredients`, `pizza_name`.
- **Variable objetivo:** `unit_price`.

## Explicación paso a paso de cada script

### 1. `analisis.py` — Exploración de datos (EDA)

Script de diagnóstico, pensado para correrse manualmente y leer los resultados por consola/gráficos. No produce ningún archivo de salida.

Qué hace, en orden:
1. **Carga** `data/pizza_sales_sin_tiempo.csv` (ya debe existir).
2. **Introduce nulos sintéticos** a propósito: marca aleatoriamente ~1% de las filas de `n_ingredientes` como `NaN`, solo para poder practicar/mostrar cómo se detectan y tratan valores faltantes (esto no se guarda, es solo para el análisis en memoria).
3. **Análisis exploratorio básico:** tipos de dato por columna, conteo de nulos, filas duplicadas, columnas constantes (candidatas a eliminar) y columnas con demasiados valores únicos (posibles IDs).
4. **Distribución de la variable objetivo** (`unit_price`): estadísticas descriptivas (`describe()`) y un histograma.
5. **Correlaciones con `unit_price`:** calcula Pearson y Spearman para todas las variables numéricas, clasifica cada una como "débil" o "conservada" según un umbral (`UMBRAL_CORRELACION = 0.1`), e imprime el detalle de las variables débiles. Grafica barras horizontales de ambas correlaciones.
6. **Importancia de variables con Random Forest:** codifica categóricas con one-hot, entrena un `RandomForestRegressor` rápido (20 árboles, profundidad 10) sobre un split 70/30, imprime el RMSE en test y el top 20 de variables más importantes con su gráfico.
7. **Detecta "variables inútiles"**: combina columnas constantes, posibles IDs y variables con importancia por debajo de `UMBRAL_IMPORTANCIA = 0.001`, y cuenta filas duplicadas o con más de 50% de nulos (`UMBRAL_NULOS`).
8. **Comparación final Regresión Lineal vs Random Forest**: entrena ambos modelos sobre el mismo split y compara RMSE/R² para concluir cuál tiene mejor desempeño (imprime el ganador).

### 2. `analisis_sin_tiempo.py` — Genera el dataset de trabajo principal

Este es el script de **feature engineering** que produce el dataset que usa todo lo demás (`main.py`, `analisis.py`, `analisis_comparativo_modelos.py`).

Qué hace, en orden:
1. **Carga** el dataset crudo `data/pizza_sales.csv`.
2. **Crea `n_ingredientes`**: cuenta cuántos ingredientes tiene cada pizza (separando el string `pizza_ingredients` por comas).
3. **One-hot encoding de ingredientes**: separa la lista de ingredientes de cada pizza, la explota en filas individuales, y genera una columna binaria (`ingrediente_<nombre>`) por cada ingrediente único que existe en todo el dataset (por eso el dataset final termina con 70 columnas: 69 features + `unit_price`).
4. **Elimina columnas irrelevantes o redundantes para el precio**: `pizza_id`, `order_id`, `pizza_name`, `pizza_name_id`, `pizza_ingredients` (ya está codificado en las columnas anteriores), `total_price` (es `unit_price × quantity`, sería fuga de información), `order_date` y `order_time` (de ahí el nombre "sin tiempo").
5. **Codifica `pizza_size` y `pizza_category`** con `LabelEncoder` (números enteros en vez de texto).
6. **Guarda el resultado** en `data/pizza_sales_sin_tiempo.csv` (avisa si ya existía, pero igual lo sobreescribe) y lo vuelve a leer desde disco.
7. **Entrena un Random Forest de prueba** (100 árboles) sobre un split 80/20 solo para reportar el RMSE y la importancia de variables por consola — este modelo **no se guarda**, es solo una validación rápida de que el dataset generado sirve para predecir bien.

### 3. `analisis_pearson_spearman.py` — Dataset filtrado por correlación

Alternativa al script anterior: en vez de quedarse con todas las columnas, filtra solo las que tienen relación estadística relevante con `unit_price`.

Qué hace, en orden:
1. **Carga** el dataset crudo `data/pizza_sales.csv`.
2. Aplica la **misma ingeniería de variables** que `analisis_sin_tiempo.py` (crea `n_ingredientes`, one-hot de ingredientes, elimina columnas irrelevantes, codifica `pizza_size`/`pizza_category` — aunque aquí con `.cat.codes` en vez de `LabelEncoder`, el resultado es equivalente).
3. **Calcula correlación de Pearson y de Spearman** de cada variable contra `unit_price` y las ordena de mayor a menor (en valor absoluto de Pearson).
4. **Filtra las variables**: se queda solo con las que tienen `|Pearson| ≥ 0.1` **o** `|Spearman| ≥ 0.1` (`UMBRAL_CORRELACION`). Esto reduce el dataset a las columnas que realmente aportan señal.
5. **Guarda el resultado filtrado** en `data/pizza_sales_pearson_spearman.csv` (mismo comportamiento de sobreescritura que el script anterior) y lo vuelve a leer desde disco.
6. **Grafica** Pearson vs Spearman lado a lado para las variables conservadas (no se guarda la imagen, solo `plt.show()`).

### 4. `analisis_comparativo_modelos.py` — Torneo de 10 modelos

Compara de forma sistemática 10 algoritmos de regresión sobre los dos datasets procesados (`pizza_sales_sin_tiempo.csv` y `pizza_sales_pearson_spearman.csv`), que **deben existir de antemano**.

Qué hace, en orden:
1. Para cada uno de los 2 datasets:
   - Carga el CSV, separa `X` (features) e `y` (`unit_price`), codifica categóricas restantes con one-hot e imputa nulos con la media.
   - Divide en train/test (80/20).
   - Para cada uno de los 10 modelos (Regresión Lineal, Ridge, Lasso, ElasticNet, Árbol de Decisión, Random Forest, Gradient Boosting, SVM-RBF, KNN, Red Neuronal MLP): arma un pipeline con `StandardScaler` + el modelo, lo entrena y calcula RMSE, MAE y R² sobre el test.
2. **Guarda todos los resultados** (20 combinaciones: 10 modelos × 2 datasets) en `resultados_comparacion.csv` (se sobreescribe cada vez).
3. **Grafica** comparaciones de RMSE y de R² por modelo y dataset (solo `plt.show()`, no se guardan las imágenes).
4. **Genera un ranking**: mejor modelo por dataset (menor RMSE), mejor dataset en promedio, mejor modelo en promedio, y un ranking completo ordenado de menor a mayor RMSE.
5. **Imprime una conclusión** con el mejor dataset, el mejor modelo y la mejor combinación específica (modelo + dataset con el RMSE más bajo de todos).

### 5. `main.py` — Entrenamiento del modelo final

Es el script "de producción": toma la decisión ya validada en los pasos anteriores (Random Forest es el mejor modelo, `pizza_sales_sin_tiempo.csv` es el dataset elegido) y entrena la versión definitiva que se guarda en disco.

Qué hace, en orden:
1. **Carga** `data/pizza_sales_sin_tiempo.csv` (debe existir de antemano; este script no lo genera).
2. **Separa** `X` (todas las columnas menos `unit_price`) e `y` (`unit_price`).
3. **Codifica categóricas restantes** con one-hot (`pd.get_dummies`) e **imputa nulos** con la media de cada columna.
4. **Divide en train/test** (80/20, `random_state=42` para reproducibilidad).
5. **Entrena un `RandomForestRegressor`** con 200 árboles (`n_estimators=200`, sin límite de profundidad).
6. **Evalúa** el modelo sobre el test: calcula e imprime RMSE, MAE y R².
7. **Guarda el modelo entrenado** con `joblib.dump(...)` en `modelo_entrenado.pkl`.
8. **Guarda el dataset con las features ya codificadas** (X + y) en `dataset_optimizado.csv`.
9. **Guarda las métricas** (dataset usado, modelo, RMSE, MAE, R²) en `metricas_finales.csv`.
10. **Calcula la importancia de cada variable** según el Random Forest, la ordena de mayor a menor, imprime el top 15, la guarda completa en `importancia_variables.csv` y genera el gráfico de barras horizontales `importancia_variables.png` (esta sí se guarda con `plt.savefig`, a diferencia de los gráficos de los otros scripts).
11. Imprime un resumen final listando los 5 archivos generados.

## Resultados

### Comparación de modelos (dataset "Pizza Sin Tiempo")

| Modelo | RMSE | MAE | R² |
|---|---|---|---|
| Regresión Lineal | 1.5027 | 0.5867 | 0.829 |
| Ridge | 1.5027 | 0.5867 | 0.829 |
| Lasso | 1.5742 | 0.7646 | 0.812 |
| ElasticNet | 1.5542 | 0.7643 | 0.817 |
| Árbol de Decisión | 0.2735 | 0.1682 | 0.994 |
| **Random Forest** | **0.0** | **0.0** | **1.0** |
| Gradient Boosting | 0.0727 | 0.0435 | 0.9996 |
| SVM (RBF) | 0.799 | 0.1743 | 0.9516 |
| KNN | 0.1408 | 0.0067 | 0.9985 |
| Red Neuronal (MLP) | 0.025 | 0.0164 | 1.0 |

Con el dataset "Pearson-Spearman" los resultados son consistentes, con Random Forest y KNN muy cerca del ajuste perfecto (R² ≈ 0.9999).

### Variables más importantes (Random Forest)

1. `pizza_size` — 78.6% de la importancia total.
2. `ingrediente_Mozzarella Cheese` — 3.9%.
3. `n_ingredientes` — 3.3%.
4. `pizza_category` — 2.5%.
5. `ingrediente_Brie Carre Cheese`, `Pears`, `Caramelized Onions`, `Prosciutto`, `Thyme`, entre otros ingredientes premium.

El tamaño de la pizza domina ampliamente la predicción del precio, seguido de la presencia de ciertos ingredientes especiales.

## Requisitos

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## Uso (orden recomendado)

```bash
# 1. Generar el dataset procesado principal (obligatorio antes de main.py)
python analisis_sin_tiempo.py

# 2. (Opcional) Generar el dataset alternativo filtrado por correlación
python analisis_pearson_spearman.py

# 3. (Opcional) Explorar el dataset con más detalle
python analisis.py

# 4. (Opcional) Comparar los 10 modelos sobre ambos datasets
python analisis_comparativo_modelos.py

# 5. Entrenar y guardar el modelo final
python main.py
```

Si ya tienes los CSV de `data/` generados de una corrida anterior (como vienen incluidos en este repo), puedes saltar directo al paso 5 y correr solo `python main.py`.

## Contexto

Proyecto desarrollado como parte del curso de Modelos y Simulación de Sistemas I.
