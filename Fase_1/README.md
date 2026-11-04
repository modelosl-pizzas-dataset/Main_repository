# Fase 1 – Predicción del precio unitario de pizzas (resumen)

Problema: predecir `unit_price` con el dataset Pizza Sales (48.620 filas) mediante regresión, usando tamaño, categoría e ingredientes. Se eligió **Random Forest**.

## Contenido de esta carpeta
- `modelo_predictivo.ipynb`: notebook de la entrega de la Fase 1. Usa un `Pipeline` con el preprocesamiento ajustado solo con train, excluye `pizza_name` y `total_price` y genera `modelo_entrenado.pkl`.
- `modelo_entrenado.pkl`: modelo guardado por el notebook (Pipeline completo).
- `analisis_sin_tiempo.py` y `main.py`: primer enfoque del equipo (ingeniería de variables con one-hot de todos los ingredientes y entrenamiento del Random Forest). Se conservan como referencia; sus salidas se pueden regenerar y se eliminaron del repositorio.

Los datos de entrada están en `../data/` (`pizza_sales.csv` y `pizza_sales_sin_tiempo.csv`). Los scripts se ejecutan desde la raíz del repositorio y el notebook desde esta carpeta.

## Nota sobre la evaluación
La partición aleatoria (`train_test_split`) deja las mismas pizzas en entrenamiento y prueba: el dataset tiene 48.620 filas pero solo 91 combinaciones distintas de pizza y tamaño, cada una con un único precio. Por eso las métricas de esta fase son optimistas: RMSE = 0.0 y R² = 1.0 en `main.py`, y R² ≈ 0.956 en el notebook. La **Fase 2** (`../Fase_2`) lo corrige con una separación por tipo de pizza y un preprocesamiento dentro de un `Pipeline`; ver su README.
