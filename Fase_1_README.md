# Fase 1 – Predicción del precio unitario de pizzas (resumen)

Problema: predecir `unit_price` con el dataset Pizza Sales (48.620 filas) mediante regresión. Tras comparar datasets y 10 algoritmos se eligió **Random Forest** sobre el dataset sin variables de tiempo.

## Código conservado
- `analisis_sin_tiempo.py`: ingeniería de variables (n.º y one-hot de ingredientes, elimina identificadores, fechas y `total_price`, codifica tamaño y categoría) → `data/pizza_sales_sin_tiempo.csv`.
- `main.py`: entrena el Random Forest final (200 árboles, semilla 42) y genera `dataset_optimizado.csv`, el modelo y las métricas.
- `modelo_predictivo.ipynb`: notebook de la entrega de la Fase 1.

## Nota
El RMSE = 0.0 reportado en esta fase se debía a una separación aleatoria con filas duplicadas (fuga de información). La **Fase 2** (`../Fase_2`) lo corrige con una separación por tipo de pizza y un preprocesamiento dentro de un `Pipeline`; ver su README.
