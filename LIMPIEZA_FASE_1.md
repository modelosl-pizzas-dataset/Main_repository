# Limpieza de la carpeta Fase_1 (antes de la Fase 2)

Criterio: conservar solo el código funcional que genera el dataset y el modelo elegido (Random Forest). Se elimina el análisis exploratorio, la comparación de datasets/modelos y las salidas que se pueden regenerar.

## Conservar
| Archivo | Motivo |
|---|---|
| `Fase_1/analisis_sin_tiempo.py` | Script de ingeniería de variables que genera el dataset elegido (en Fase 2 su lógica pasó a `prepare_data.py` + `src/features.py`). |
| `Fase_1/main.py` | Entrena el Random Forest final y genera `dataset_optimizado.csv`. |
| `Fase_1/README.md` | Reemplazar por el `Fase_1_README.md` corto incluido en este paquete. |
| `Fase_1/modelo_predictivo.ipynb` | Entregable formal de la Fase 1 (la Fase 2 pide conservar `fase-1`). **Decisión del grupo:** se deja; puede borrarse si el docente no lo exige. |
| `data/pizza_sales.csv` y `data/pizza_sales_sin_tiempo.csv` | Entradas que necesitan los dos scripts conservados. |

## Eliminar
| Archivo | Por qué sobra |
|---|---|
| `Fase_1/analisis.py` | EDA y selección de variables (no genera nada). |
| `Fase_1/analisis_pearson_spearman.py` y `data/pizza_sales_pearson_spearman.csv` | Dataset alternativo descartado. |
| `Fase_1/analisis_comparativo_modelos.py` y `Fase_1/resultados_comparacion.csv` | Torneo de 10 modelos; ya se eligió Random Forest. |
| `Fase_1/README_model.md` | Documentación redundante de la comparación. |
| `Fase_1/modelo_entrenado.pkl`, `metricas_finales.csv`, `importancia_variables.csv`, `importancia_variables.png`, `dataset_optimizado.csv` | Salidas regenerables de `main.py` (su RMSE = 0 era resultado de la fuga explicada en el README de Fase_2; `dataset_optimizado.csv` de Fase 1 era además copia del dataset sin tiempo). |
| `.idea/` | Configuración del IDE (agregar `.idea/` al `.gitignore`). |

## Comandos (Git Bash / Bash, desde la raíz del repositorio; en una rama `feature/limpieza-fase1`)
```
git checkout develop
git checkout -b feature/limpieza-fase1
git rm Fase_1/analisis.py Fase_1/analisis_pearson_spearman.py Fase_1/analisis_comparativo_modelos.py Fase_1/README_model.md
git rm Fase_1/resultados_comparacion.csv Fase_1/modelo_entrenado.pkl Fase_1/metricas_finales.csv Fase_1/importancia_variables.csv Fase_1/importancia_variables.png Fase_1/dataset_optimizado.csv
git rm data/pizza_sales_pearson_spearman.csv
git rm -r --cached .idea
cp Fase_1_README.md Fase_1/README.md
git add Fase_1 && git commit -m "Limpieza de Fase 1: se conserva solo el codigo que genera el dataset y el modelo final"
```
(Si no existe la rama `develop`, créela desde `main` primero. En PowerShell, `git rm` es igual; reemplace `cp` por `Copy-Item Fase_1_README.md Fase_1\README.md -Force`.)
