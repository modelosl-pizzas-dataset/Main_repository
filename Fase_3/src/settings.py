"""Configuración de la API (variables de entorno con valores por defecto)."""
import os
import re
from dataclasses import dataclass
from pathlib import Path

from src.config import GROUP_COLUMN, TARGET

_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class Settings:
    """Parámetros de ejecución de la API.

    Variable de entorno      Default                          Uso
    MODEL_VERSION            v1                               Versión que atiende /predict
    MODELS_DIR               models                           Carpeta con las versiones (models/<version>/)
    DATA_PATH                data/dataset_optimizado.csv      Datos con los que entrena /train
    TARGET                   unit_price                       Variable objetivo de /train
    GROUP_COLUMN             pizza_name                       Grupo para separar train/test ('none' = aleatorio)
    TRAIN_SEED               42                               Semilla por defecto de /train
    TEST_SIZE                0.2                              Fracción de prueba de /train
    CORS_ORIGINS             *                                Origen permitido para un futuro frontend ('' = desactivado)

    Las rutas relativas se resuelven respecto a la carpeta de api.py (en Docker, /app).
    """
    models_dir: Path
    model_version: str
    data_path: Path
    target: str = TARGET
    group_column: str = GROUP_COLUMN
    seed: int = 42
    test_size: float = 0.2
    cors_origins: str = "*"

    @classmethod
    def from_env(cls, base_dir, env=None):
        env = os.environ if env is None else env
        base_dir = Path(base_dir)

        def path(name, default):
            p = Path(env.get(name, default))
            return p if p.is_absolute() else base_dir / p

        version = env.get("MODEL_VERSION", "v1").strip()
        if not _VERSION_RE.match(version):
            raise ValueError(f"MODEL_VERSION no válida: '{version}'")
        try:
            seed = int(env.get("TRAIN_SEED", "42"))
            test_size = float(env.get("TEST_SIZE", "0.2"))
        except ValueError as exc:
            raise ValueError(f"TRAIN_SEED/TEST_SIZE deben ser numéricos: {exc}") from exc
        group = env.get("GROUP_COLUMN", GROUP_COLUMN).strip()
        return cls(
            models_dir=path("MODELS_DIR", "models"),
            model_version=version,
            data_path=path("DATA_PATH", "data/dataset_optimizado.csv"),
            target=env.get("TARGET", TARGET),
            group_column=None if group.lower() in ("", "none") else group,
            seed=seed,
            test_size=test_size,
            cors_origins=env.get("CORS_ORIGINS", "*"),
        )
