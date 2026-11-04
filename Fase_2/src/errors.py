"""Errores previsibles de la aplicación (se reportan con mensaje claro y código de salida != 0)."""


class AppError(Exception):
    """Error previsible: el CLI muestra el mensaje y termina con código de salida 1."""


class DataValidationError(AppError):
    """Los datos de entrada no cumplen el esquema esperado."""


class ModelNotFoundError(AppError):
    """No existe (o no se puede leer) el modelo solicitado."""


class VersionExistsError(AppError):
    """El identificador de versión ya existe; no se sobrescribe."""
