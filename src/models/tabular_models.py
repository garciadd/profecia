from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor


DEFAULT_RANDOM_STATE = 42


def build_model(
    model_name: str,
    random_state: int = DEFAULT_RANDOM_STATE,
    **kwargs,
):
    """
    Crea un modelo tabular de regresión a partir de su nombre.

    Modelos soportados
    ------------------
    - "rf"  : RandomForestRegressor
    - "hgb" : HistGradientBoostingRegressor
    - "mlp" : MLPRegressor

    Parameters
    ----------
    model_name : str
        Nombre corto del modelo.
    random_state : int
        Semilla para reproducibilidad.
    **kwargs
        Hiperparámetros adicionales específicos del modelo.

    Returns
    -------
    model
        Instancia del modelo sklearn correspondiente.
    """
    model_name = model_name.lower().strip()

    if model_name == "rf":
        default_params = {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "n_jobs": -1,
            "random_state": random_state,
        }
        default_params.update(kwargs)
        return RandomForestRegressor(**default_params)

    if model_name == "hgb":
        default_params = {
            "loss": "squared_error",
            "learning_rate": 0.1,
            "max_iter": 200,
            "max_depth": None,
            "min_samples_leaf": 20,
            "random_state": random_state,
        }
        default_params.update(kwargs)
        return HistGradientBoostingRegressor(**default_params)

    if model_name == "mlp":
        default_params = {
            "hidden_layer_sizes": (128, 64),
            "activation": "relu",
            "solver": "adam",
            "alpha": 1e-4,
            "batch_size": "auto",
            "learning_rate": "adaptive",
            "learning_rate_init": 1e-3,
            "max_iter": 300,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "random_state": random_state,
        }
        default_params.update(kwargs)
        return MLPRegressor(**default_params)

    raise ValueError(
        f"Modelo no soportado: '{model_name}'. "
        "Usa uno de: 'rf', 'hgb', 'mlp'."
    )


def get_model_name(model) -> str:
    """
    Devuelve el nombre de clase del modelo.
    """
    return model.__class__.__name__


def get_default_model_params(model_name: str) -> dict:
    """
    Devuelve los hiperparámetros por defecto del modelo solicitado.
    Útil para inspección o logging.
    """
    model = build_model(model_name)
    return model.get_params()