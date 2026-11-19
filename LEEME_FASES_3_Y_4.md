# Cómo agregar las Fases 3 y 4 al proyecto

1. Copie las carpetas `Fase_3` y `Fase_4` a la raíz del repositorio, junto a `Fase_1` y `Fase_2`.
2. Cada carpeta es **autónoma** (incluye su `src/`, datos de entrenamiento `data/dataset_optimizado.csv` y los modelos `v1` y `v2`), de modo que se construye con Docker desde su propia carpeta.
3. Ejecute cada fase desde su carpeta (los comandos están en su `README.md` y `DOCKER_PASOS.txt`).
4. Complete en los README los placeholders marcados con _(completar)_ y pegue su propia evidencia de ejecución.

## Git Flow sugerido (desde `develop`)
```
git checkout develop
git checkout -b feature/api-fase3
git add Fase_3 && git commit -m "Fase 3: API REST (health, predict, train) con pruebas, cliente y Dockerfile"
git push -u origin feature/api-fase3          # Pull Request hacia develop, revisado por otro integrante

git checkout develop
git checkout -b feature/monitoreo-fase4
git add Fase_4 && git commit -m "Fase 4: registro de solicitudes, script de monitoreo, politica de reentrenamiento y reporte"
git push -u origin feature/monitoreo-fase4    # Pull Request hacia develop
```
Al final: integrar `develop` en `main` y crear la etiqueta (`git tag entrega-3-4`).
