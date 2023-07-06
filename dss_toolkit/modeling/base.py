import dss_toolkit.modeling.algorithms as algorithms_list
from sklearn.base import BaseEstimator


class BaseModel(BaseEstimator):
    def __init__(self, algorithm=None, model_args={}, fit_args={}):
        self.algorithm = algorithm
        self.model_args = model_args
        self.fit_args = fit_args
        self.model_ = None

    def fit(self, X, y=None, **kwargs):
        return self.model_.fit(X, y, **self.fit_args)

    def predict_proba(self, X, **kwargs):
        invert_op = getattr(self, "predict_proba", None)
        if invert_op is not None and callable(invert_op):
            return self.model_.predict_proba(X, **kwargs)  # Probablity for all classes
        else:
            raise Exception(f"No method `predict_proba` for instance of {self.model__}")

    def predict(self, X, **kwargs):
        return self.model_.predict(X, **kwargs)

    def get_algorithm_class(self, task, algorithm_name):
        if type(algorithm_name) != str:
            return algorithm_name

        if task == "classification":
            available_algorithms = algorithms_list.CLASSIFICATION
        elif task == "regression":
            available_algorithms = algorithms_list.REGRESSION
        elif task == "clustering":
            available_algorithms = algorithms_list.CLUSTERING
        else:
            available_algorithms = dict()

        model_class = available_algorithms.get(algorithm_name, None)

        if model_class is None:
            error_msg = f"""Algorithm parameter `{algorithm_name}` is not available from list of {task} algorithms
            Available:{list(available_algorithms.keys())}))"""
            raise Exception(error_msg)

        return model_class

    def get_algorithm_list(task):
        if task == "classification":
            available_algorithms = algorithms_list.CLASSIFICATION
        elif task == "regression":
            available_algorithms = algorithms_list.REGRESSION
        elif task == "clustering":
            available_algorithms = algorithms_list.CLUSTERING
        else:
            available_algorithms = dict()
        return list(available_algorithms.keys())

    def evaluate(self, X, y, eval_func, use_proba=False, **kwargs):
        if use_proba:
            y_proba = self.predict_proba(X)
            return eval_func(y, y_proba, **kwargs)
        else:
            y_pred = self.predict(X)
            return eval_func(y, y_pred, **kwargs)


class BinaryClassifierModel(BaseModel):
    def __init__(self, algorithm=None, model_args={}, fit_args={}):
        super(BinaryClassifierModel, self).__init__(algorithm, model_args, fit_args)

        # Get model class from mapping or treat as the model class itself
        model_class = self.get_algorithm_class("classification", algorithm)
        # Construct model class
        self.model_ = model_class(**model_args)

    def predict_proba(self, X, **kwargs):
        proba_classes = self.model_.predict_proba(X, **kwargs)
        return proba_classes[:, 1]  # Probabilty of class 1

    def available_algorthms(names_only=True):
        algo_list = algorithms_list.CLASSIFICATION
        if names_only:
            return list(algo_list.keys())

        return algo_list

    def add_algorithm(name, constructor):
        algorithms_list.CLASSIFICATION.update({name: constructor})


class RegressionModel(BaseModel):
    def __init__(self, algorithm=None, model_args={}, fit_args={}):
        super(RegressionModel, self).__init__(algorithm, model_args, fit_args)

        # Get model class from mapping or treat as the model class itself
        model_class = self.get_algorithm_class("regression", algorithm)
        # Construct model class
        self.model_ = model_class(**model_args)

    def available_algorthms(names_only=True):
        algo_list = algorithms_list.REGRESSION
        if names_only:
            return list(algo_list.keys())

        return algo_list

    def add_algorithm(name, constructor):
        algorithms_list.REGRESSION.update({name: constructor})


class UnsupervisedModel(BaseModel):
    def predict(self, X, **kwargs):
        # check if predict() is available, use fit_predict if Not available
        invert_op = getattr(self.model_, "predict", None)
        if invert_op is not None and callable(invert_op):
            result = self.model_.predict(X, **kwargs)
        else:
            result = self.model_.fit_predict(X)
        return result

    def evaluate(self, X, eval_func, **kwargs):
        result = self.predict(X, **kwargs)

        return eval_func(X, result)


class ClusteringModel(UnsupervisedModel):
    def __init__(self, algorithm=None, model_args={}, fit_args={}):
        super(ClusteringModel, self).__init__(algorithm, model_args, fit_args)

        # Get model class from mapping or treat as the model class itself
        model_class = self.get_algorithm_class("clustering", algorithm)
        # Construct model class
        self.model_ = model_class(**model_args)

    def available_algorthms(names_only=True):
        algo_list = algorithms_list.CLUSTERING
        if names_only:
            return list(algo_list.keys())

        return algo_list

    def add_algorithm(name, constructor):
        algorithms_list.CLUSTERING.update({name: constructor})
