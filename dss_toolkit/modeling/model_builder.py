class ModelBuilder:
    def __init__(self, task, algorithm, hyperparameters=dict):

        model = ModelInstance()  # Generic Model placeholder

        if algorithm == "logistic_regresion":
            pass
        elif algorithm == "xgboost":
            pass
        else:
            pass

        return model


class ModelInstance:
    """
    Generic Model Wrapper for different model libraries
    """

    def __init__(self, model):
        self.model

    def fit(self, X, y, validation_data=None, validation_split=None):

        return self.model.fit(X, y,)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        y_pred = self.model.predict(X)

        metrics = evaluate_metrics(y, y_pred)  # Replace with classification/regression/clustering metric
        return metrics


def evaluate_metrics(y_true, y_pred):
    pass
