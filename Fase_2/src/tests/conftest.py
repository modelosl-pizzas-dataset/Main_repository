import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PIZZAS = {  # tipo -> (categoría, ingredientes, precio base en talla S)
    "Hawaiana": ("Classic", "Ham, Pineapple, Cheese", 10.0),
    "Pepperoni": ("Classic", "Pepperoni, Cheese", 9.5),
    "Veggie": ("Veggie", "Mushrooms, Tomatoes, Spinach, Garlic", 11.0),
    "Greek": ("Veggie", "Feta, Olives, Tomatoes", 11.5),
    "Pollo BBQ": ("Chicken", "Chicken, Onions, BBQ Sauce", 12.0),
    "Pollo Pesto": ("Chicken", "Chicken, Pesto, Tomatoes", 12.5),
    "Suprema": ("Supreme", "Sausage, Peppers, Onions, Olives, Cheese", 13.0),
    "Picante": ("Supreme", "Salami, Jalapeno, Peppers", 12.8),
}
SIZE_EXTRA = {"S": 0.0, "M": 4.0, "L": 8.0}


@pytest.fixture
def synthetic_df():
    """Dataset pequeño y sintético (sin descargas) con el mismo esquema que dataset_optimizado.csv."""
    rng = np.random.default_rng(0)
    rows = []
    for name, (cat, ingr, base) in PIZZAS.items():
        for size, extra in SIZE_EXTRA.items():
            for _ in range(15):
                rows.append({"pizza_name": name, "pizza_size": size, "pizza_category": cat,
                             "pizza_ingredients": ingr, "unit_price": base + extra})
    df = pd.DataFrame(rows).sample(frac=1, random_state=int(rng.integers(100))).reset_index(drop=True)
    return df


@pytest.fixture
def synthetic_csv(tmp_path, synthetic_df):
    path = tmp_path / "datos.csv"
    synthetic_df.to_csv(path, index=False)
    return path


@pytest.fixture
def new_data():
    """Datos nuevos SIN variable objetivo."""
    return pd.DataFrame({
        "pizza_size": ["S", "M", "L"],
        "pizza_category": ["Classic", "Veggie", "Chicken"],
        "pizza_ingredients": ["Ham, Cheese", "Mushrooms, Spinach", "Chicken, Onions, Pesto"],
    })
