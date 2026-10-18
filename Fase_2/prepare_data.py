"""Genera data/dataset_optimizado.csv a partir del CSV crudo (pizza_sales.csv).

Es la versión de Fase 2 del script de ingeniería de variables de Fase 1 (analisis_sin_tiempo.py):
aquí solo se hacen transformaciones SIN estado; el vocabulario de ingredientes y las
codificaciones se aprenden dentro del Pipeline, únicamente con los datos de entrenamiento.
"""
import argparse
import sys

from src.data import prepare_dataset, read_csv
from src.errors import AppError


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Genera el dataset optimizado a partir del CSV crudo de Pizza Sales.")
    p.add_argument("--raw", default="data/pizza_sales.csv", help="CSV crudo (default: data/pizza_sales.csv)")
    p.add_argument("--output", default="data/dataset_optimizado.csv", help="CSV de salida (default: data/dataset_optimizado.csv)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        df = prepare_dataset(read_csv(args.raw, "CSV crudo"))
        df.to_csv(args.output, index=False)
    except AppError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Dataset optimizado guardado en {args.output}: {df.shape[0]} filas, {df.shape[1]} columnas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
