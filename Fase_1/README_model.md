# Fase 1 - Modelo Predictivo

## Objetivo

Construir y evaluar un modelo de Machine Learning capaz de predecir el precio unitario (`unit_price`) de las pizzas.

## Dataset

El dataset utilizado es `pizza_sales.csv`, ubicado en la carpeta `data/`.

## Preparación de los datos

Se realizaron las siguientes transformaciones:

- Se creó `n_ingredientes`, que representa la cantidad de ingredientes.
- Se crearon variables binarias para identificar la presencia de pollo, carnes y vegetales.
- Se eliminaron variables identificadoras que no aportan información generalizable.
- Se eliminó `total_price` porque está directamente relacionado con `unit_price` y `quantity`, evitando fuga de información.
- Se eliminaron las variables de fecha y hora.
- Se excluyó `pizza_name` para evitar que el modelo memorice el precio específico asociado a cada producto.

## Modelo

Se utilizó `RandomForestRegressor`.

El preprocesamiento se implementó mediante un `Pipeline` de Scikit-learn con:

- Imputación de valores faltantes numéricos mediante la mediana.
- Imputación de valores faltantes categóricos mediante la categoría más frecuente.
- Codificación One-Hot de variables categóricas.
- Entrenamiento del modelo Random Forest.

Los datos se dividieron en 80% para entrenamiento y 20% para prueba, utilizando `random_state=42`.

## Evaluación

Resultados obtenidos sobre el conjunto de prueba:

| Métrica | Resultado |
|---|---:|
| RMSE | 0.763985 |
| MAE | 0.135845 |
| R² | 0.955790 |

## Prevención de fuga de información

Se eliminó `total_price` debido a su relación directa con `unit_price`.

Además, el preprocesamiento está dentro de un `Pipeline`, por lo que los transformadores se ajustan únicamente con los datos de entrenamiento.

También se excluyó `pizza_name` porque identifica de manera muy específica el producto y podría favorecer la memorización de precios.

## Archivos

- `modelo_predictivo.ipynb`: notebook con la exploración, preparación, entrenamiento y evaluación.
- `modelo_entrenado.pkl`: modelo entrenado y guardado mediante Joblib.

## Ejecución

1. Abrir `modelo_predictivo.ipynb`.
2. Verificar que el dataset se encuentre en `../data/pizza_sales.csv`.
3. Ejecutar las celdas del notebook en orden.
4. El modelo entrenado se guarda como `modelo_entrenado.pkl`.

## Conclusión

Se construyó un modelo predictivo utilizando Random Forest para estimar `unit_price`. El modelo fue evaluado mediante RMSE, MAE y R² y se implementó un pipeline de preprocesamiento para mantener separado el procesamiento de los datos de entrenamiento y prueba.
