from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
ARCHIVO_CSV = BASE_DIR / "data" / "pizza_sales.csv"

df = pd.read_csv(ARCHIVO_CSV)

df["n_ingredientes"] = df["pizza_ingredients"].str.split(",").apply(len)

df = df.drop(columns=[
    "pizza_id",
    "order_id",
    "pizza_name",
    "pizza_name_id",
    "pizza_ingredients",
    "total_price",
    "order_date",
    "order_time"
])

for c in ["pizza_size", "pizza_category"]:
    df[c] = df[c].astype("category").cat.codes

pearson = df.corr(method="pearson")["unit_price"]
spearman = df.corr(method="spearman")["unit_price"]

comparacion = pd.DataFrame({
    "Pearson": pearson,
    "Spearman": spearman
})

comparacion = comparacion.sort_values(
    by="Pearson",
    key=abs,
    ascending=False
)

print("\nCorrelaciones con unit_price")
print(comparacion)

comparacion.plot(kind="bar", figsize=(10,5))
plt.title("Pearson vs Spearman")
plt.grid()
plt.tight_layout()
plt.show()