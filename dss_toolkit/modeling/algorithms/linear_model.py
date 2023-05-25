from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    SGDRegressor,
    Perceptron,
)


def train_linear_regression(train_data, train_targets, test_data=None, test_targets=None, *args, **kwargs):
    model_type = kwargs.get("regression_type", "simple")

    models = {
        "simple": LinearRegression(),
        "ridge": Ridge(),
        "lasso": Lasso(),
        "elasticnet": ElasticNet(),
        "sgd": SGDRegressor(),
        "perceptron": Perceptron(),
    }

    lr_model = models.get(model_type)
    lr_model.fit(X=train_data, y=train_targets)
    return lr_model, None


def predict_linear_regression(model, X, y=None):
    return model.predict(X)
