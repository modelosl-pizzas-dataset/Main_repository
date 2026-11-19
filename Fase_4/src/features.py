"""Preprocesamiento reutilizable: se ajusta SOLO con entrenamiento y viaja dentro del Pipeline guardado."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.config import SIZE_ORDER


def split_ingredients(value):
    """Convierte 'A, B, C' en ['A', 'B', 'C']. Un nulo devuelve None."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return None
    return [x.strip() for x in str(value).split(",") if x.strip()]


class IngredientVectorizer(BaseEstimator, TransformerMixin):
    """Genera n_ingredientes + una columna binaria por ingrediente.

    - fit: aprende el vocabulario de ingredientes únicamente con los datos de entrenamiento.
    - transform: ingredientes no vistos se ignoran; un valor nulo produce ceros y
      n_ingredientes = NaN (que luego imputa SimpleImputer con la mediana aprendida).
    """

    def fit(self, X, y=None):
        values = self._as_series(X)
        vocab = set()
        for v in values:
            items = split_ingredients(v)
            if items:
                vocab.update(items)
        self.vocabulary_ = sorted(vocab)
        self._index_ = {name: i for i, name in enumerate(self.vocabulary_)}
        return self

    @staticmethod
    def _as_series(X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0].tolist()
        return list(np.asarray(X).ravel())

    def transform(self, X):
        values = self._as_series(X)
        out = np.zeros((len(values), len(self.vocabulary_) + 1), dtype=float)
        for row, v in enumerate(values):
            items = split_ingredients(v)
            if items is None:
                out[row, 0] = np.nan
                continue
            out[row, 0] = len(items)
            for item in items:
                col = self._index_.get(item)
                if col is not None:
                    out[row, col + 1] = 1.0
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(["n_ingredientes"] + [f"ingrediente_{i}" for i in self.vocabulary_], dtype=object)


def build_preprocessor():
    """ColumnTransformer sin ajustar: tamaño (ordinal), categoría (one-hot) e ingredientes."""
    size = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=[SIZE_ORDER])),
    ])
    category = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    ingredients = Pipeline([
        ("vectorizer", IngredientVectorizer()),
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ])
    return ColumnTransformer(
        [
            ("tamano", size, ["pizza_size"]),
            ("categoria", category, ["pizza_category"]),
            ("ingredientes", ingredients, ["pizza_ingredients"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _clean_cell(value):
    """Quita espacios; una cadena vacía se trata como nulo."""
    if isinstance(value, str):
        value = value.strip()
        return value if value else np.nan
    return value


def to_model_frame(df, columns):
    """Selecciona las columnas del esquema EN EL ORDEN DE ENTRENAMIENTO y normaliza los nulos a np.nan."""
    out = df[list(columns)].copy()
    for col in out.columns:
        s = out[col].astype(object)
        out[col] = s.where(s.notna(), np.nan)
        out[col] = out[col].map(_clean_cell)
    return out
