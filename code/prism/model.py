import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


def train_regressor(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_seed: int,
    mini_mode: bool = False,
) -> RegressorMixin:
    if mini_mode:
        model = LinearRegression()
    else:
        model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=random_seed,
            n_jobs=-1,
        )

    model.fit(x_train, y_train)
    return model
