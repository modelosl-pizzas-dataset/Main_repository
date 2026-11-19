"""Punto de entrada de predicción. Ejemplo (una sola línea):

python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
"""
import argparse
import sys

from src.errors import AppError
from src.prediction import predict_file


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Genera predicciones de unit_price con una versión guardada (no reentrena).")
    p.add_argument("--model", required=True, help="Ruta a modelo.joblib o a la carpeta de la versión (p. ej. models/v1)")
    p.add_argument("--input", required=True, help="CSV con pizza_size, pizza_category y pizza_ingredients")
    p.add_argument("--output", required=True, help="CSV de salida (entrada + columna 'prediccion')")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        info = predict_file(args.model, args.input, args.output)
    except AppError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # error no previsto: mensaje claro y código != 0, sin archivo de salida
        print(f"ERROR inesperado: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if info["extra_columns"]:
        print(f"Aviso: columnas adicionales ignoradas por el modelo: {info['extra_columns']}")
    print(f"Versión del modelo: {info['version']} | filas predichas: {info['n_rows']}")
    print(f"Predicciones guardadas en: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
