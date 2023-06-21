import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


class MLAlgorithms:
    """
    Static class to contain available algorithms
    """

    classification_algorithms = {
        "xgboost": xgb.XGBClassifier,
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "k_neighbors": KNeighborsClassifier,
    }

    @classmethod
    def set_defaults(cls):
        cls.classification_algorithms = {
            "xgboost": xgb.XGBClassifier,
            "logistic_regression": LogisticRegression,
            "random_forest": RandomForestClassifier,
            "k_neighbors": KNeighborsClassifier,
        }

    @classmethod
    def get_available_algorithms(cls):
        return list(cls.classification_algorithms.keys())

    @classmethod
    def add_algorithm(cls, category, name, algo_class):
        if category == "classification":
            cls.classification_algorithms.update({name: algo_class})


class BaseModel:
    def __init__(self, algorithm=None, model_args={}, fit_args={}):
        self.algorithm = algorithm
        self.model_args = model_args
        self.fit_args = fit_args

        # Get model class from mapping or treat as the model class itself
        model_class = MLAlgorithms.classification_algorithms.get(algorithm, algorithm)

        if model_class is None:
            error_msg = f"Algorithm parameter `{algorithm}` is not available from `algorithm_class_mapping` variable{list(MLAlgorithms.classification_algorithms.keys())}"
            raise Exception(error_msg)

        # Construct model class
        self.model_ = model_class(**model_args)

    def fit(self, X, y, **kwargs):
        return self.model_.fit(X, y, **self.fit_args)

    def predict_proba(self, X, **kwargs):
        invert_op = getattr(self, "predict_proba", None)
        if invert_op is not None and callable(invert_op):
            return self.model_.predict_proba(X, **kwargs)  # Probablity for all classes
        else:
            raise Exception(f"No method `predict_proba` for instance of {self.model__}")

    def predict(self, X, **kwargs):
        return self.model_.predict(X, **kwargs)


class BinaryClassifier(BaseModel):
    def predict_proba(self, X, **kwargs):
        proba_classes = self.model_.predict_proba(X, **kwargs)
        return proba_classes[:, 1]  # Probabilty of class 1
