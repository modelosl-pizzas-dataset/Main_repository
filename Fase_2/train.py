"""Punto de entrada de entrenamiento. Ejemplo (una sola línea):

python train.py --data data/dataset_optimizado.csv --target unit_price --output-dir models --version v1 --seed 42
"""
import argparse
import sys

from src.config import GROUP_COLUMN, TARGET
from src.errors import AppError
from src.training import train


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Entrena un Random Forest para predecir unit_price, lo evalúa contra un modelo base "
                    "y guarda una versión (modelo + preprocesamiento + métricas + metadatos).")
    p.add_argument("--data", required=True, help="CSV de datos etiquetados (p. ej. data/dataset_optimizado.csv)")
    p.add_argument("--target", default=TARGET, help=f"Variable objetivo (default: {TARGET})")
    p.add_argument("--output-dir", default="models", help="Directorio donde se crean las versiones (default: models)")
    p.add_argument("--version", required=True, help="Identificador único de la versión (p. ej. v1). No se sobrescribe.")
    p.add_argument("--seed", type=int, default=42, help="Semilla aleatoria (default: 42)")
    p.add_argument("--test-size", type=float, default=0.2, help="Fracción para prueba (default: 0.2)")
    p.add_argument("--group-column", default=GROUP_COLUMN,
                   help=f"Columna de grupo para separar train/test sin fuga (default: {GROUP_COLUMN}). "
                        "Use 'none' para una separación aleatoria.")
    p.add_argument("--n-estimators", type=int, default=None, help="Número de árboles (default: 200)")
    p.add_argument("--max-depth", type=int, default=None, help="Profundidad máxima (default: sin límite)")
    p.add_argument("--git-commit", default=None, help="Hash del commit (útil en Docker, donde no hay .git)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    group = None if args.group_column.lower() == "none" else args.group_column
    try:
        res = train(args.data, args.target, args.output_dir, args.version, seed=args.seed,
                    test_size=args.test_size, group_column=group, n_estimators=args.n_estimators,
                    max_depth=args.max_depth, git_commit=args.git_commit)
    except AppError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    m, md = res["metrics"], res["metadata"]
    print("=" * 60)
    print(f"Versión {md['version']} entrenada | train={md['split']['n_train']} test={md['split']['n_test']} | {md['split']['strategy']}")
    print(f"{'Modelo':<24}{'RMSE':>9}{'MAE':>9}{'R2':>9}")
    for key, label in [("modelo", "Random Forest"), ("modelo_base", "Regresión lineal (base)"),
                       ("referencia_media", "Media (referencia)")]:
        print(f"{label:<24}{m[key]['rmse']:>9.4f}{m[key]['mae']:>9.4f}{m[key]['r2']:>9.4f}")
    print("Archivos generados:")
    for name, path in res["paths"].items():
        print(f"  - {name}: {path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
