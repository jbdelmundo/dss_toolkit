import sklearn
import optuna
import warnings
from dss_toolkit.modeling.base import BinaryClassifierModel

BINARY_METRICS_MAPPING = {
    "accuracy": sklearn.metrics.accuracy_score,
    "precision": sklearn.metrics.precision_score,
    "recall": sklearn.metrics.recall_score,
    "f1": sklearn.metrics.f1_score,
    "roc_auc": sklearn.metrics.roc_auc_score,
}


class HyperParameterOptimizer:
    def __init__(
        self,
        model_creator,
        search_function,
        task="superivsed",
        predict_method="predict",
        metric=None,
        metric_args={},
        n_trials=10,
        direction="maximize",
    ):
        self.model_creator = model_creator
        self.search_function = search_function
        self.task = task
        self.predict_method = predict_method
        self.n_trials = n_trials
        self.metric = metric
        self.metric_args = metric_args
        self.direction = direction

        self.study = None
        """
        Creates an object for the hyperparameter optmiziation using Optuna

        Parameters
        ------------
        model_creator: callable
            Used to initailize the model `model_creator(**hyperparameters)`
            Returns a model with `fit()` and `predict()` function. For classification models, `predict_proba()` can be available
            For unsupervised_learning, `fit_predict()` can be used
        search_function: callable
            Function that accepts `optuna.trial` object and returns a tuple `(model_creator, hyperparamters)`
            `model_creator` is a callable to create a `model_creator(**hyperparameters)`
            `hyperparameters` is a dictionary for the model arguments        
        task: str
            `supervised`: calls metric(y_true,y_pred) or 
            `unsupervised`: calls metric(X, labels)
        predict_method: str
            How the model output will be generated.
            `'predict'` calls `model.predict()`, `predict_proba()'` calls `model.predict_proba()`,
            `'fit_predict'` calls `model.fit_predict()`
        metric: callable or str
            Function that evaluates the model output. 
                For supervised learning, returns `metric(y_true, y_pred, ** metric_args)`
                For unsupervised learning returns `metric(X, labels, ** metric_args)`
        n_trials: int
            Number of trials (set of hyperparameters) to try
        """

    def search(self, X, y=None, validation_data=None):
        """
        Creates optuna study
        Creates custom objective function
        Pass the custom objective function on the study.optimize() functiom
        """

        # Create Optuna Study, specify if minimize or maximize the metric
        study = optuna.create_study(direction=self.direction)

        # Wrap the objective function to receive args
        def objective_func_wrapper(trial):
            return self._objective_function(
                trial,
                model_creator=self.model_creator,
                hyperparameter_selector_function=self.search_function,
                task=self.task,
                predict_method=self.predict_method,
                X=X,
                y=y,
                validation_data=validation_data,
                metric_function=self.metric,
                metric_args=self.metric_args,
            )

        # Run objective function multiple times to search for the best trial
        study.optimize(objective_func_wrapper, n_trials=self.n_trials)

        self.study = study

        return study

    def _objective_function(
        self,
        trial,
        model_creator,
        hyperparameter_selector_function,
        task="supervised",
        predict_method="predict",
        X=None,
        y=None,
        validation_data=None,
        metric_function=None,
        metric_args=dict(),
    ):
        """
        Private function to do the following:
        1. Return a hyperparameter set from the selector function
        2. Train and evaluate the model using training or validation data (if given)
        3. Return metric (float)
        This method is called for each trial


        Parameters
        ----------
        trial: trial object from optuna
        hyperparameter_selector_function: callable
            Returns str, dict
        X: pd.DataFrame
        y: pd.Series or np.ndarray
        validation data: tuple
            Validation data in form of (X, y)
        metric: str or callable
            Receives two arguments(y_true and y_pred), should return a number
            if `validation_data` is provided, returns calulation on validation data
        score_on_proba: bool:
            Computes metric on `y_true` VS `y_proba` instead of `y_true` vs `y_pred`


        Returns
        -------
        metric: float
        model score to minimize/maximize
        """

        # Hyperparameters
        h_algorithm, model_hyperparameters = hyperparameter_selector_function(trial)

        if h_algorithm:
            # if model creator supports selection of algorithm
            hyper_model = model_creator(algorithm=h_algorithm, model_args=model_hyperparameters)
        else:
            hyper_model = model_creator(**model_hyperparameters)

        # Checks if validation data is provided, uses X, Y otherwise
        X_val, y_val = _default_validation_data(X, y, validation_data)

        # Train and predict
        fitted_model, val_preds = _fit_predict_model(hyper_model, X, y, validation_data, predict_method)

        # Evaluate model
        if task == "supervised":
            _compare_actuals_vs_prediction_dims(y_val, val_preds)
            model_score = metric_function(y_val, val_preds, **metric_args)

        elif task == "unsupervised":
            _compare_x_dims(X_val, val_preds)
            model_score = metric_function(X_val, val_preds, **metric_args)
        else:
            raise ValueError(f"Unknown task paramter: {task}")

        return model_score


def _fit_predict_model(model, X, y, validation_data, predict_method):
    predict_method = _verify_predict_method_available(model, predict_method)
    X_val, y_val = _default_validation_data(X, y, validation_data)  # Returns X,y if no validation is given

    # For fit_predict() use validation data-- to be consistent with predictions
    if predict_method == "fit_predict":
        if y_val is None:
            preds = model.fit_predict(X_val)
        else:
            preds = model.fit_predict(X_val, y_val)
        return model, preds

    if y is None:
        model.fit(X)  # Unsuperivsed model
    else:
        model.fit(X, y)  # Supervised model

    # Predict on Validation data (or `X`` if validation data is None)
    if predict_method == "predict_proba":
        preds = model.predict_proba(X_val)
    else:
        preds = model.predict(X_val)

    return model, preds


def _verify_predict_method_available(model, predict_method):
    """Verify if model has the predict method (`predict()` or  `predict_proba()`)"""
    has_predict_function = callable(getattr(model, predict_method, None))

    if predict_method == "predict_proba" and not has_predict_function:
        raise NotImplementedError(f"`predict_proba()` method is not available for {str(model)}")

    if predict_method == "predict" and not has_predict_function:
        warnings.warn(f"`predict()` method is not available for {str(model)}. Using `fit_predict()` instead")
        predict_method = "fit_predict"

    return predict_method


def _default_validation_data(X, y, validation_data):
    """Returns original X,y if validation is None, else returns validation data tuple"""
    if not validation_data:
        return X, y  # Return original X and y as default
    if y is None:
        X_val, y_val = validation_data, None
    else:
        X_val, y_val = validation_data

    return X_val, y_val


def _compare_actuals_vs_prediction_dims(actuals, prediction):
    if actuals.shape != prediction.shape:
        msg = f"Model output dimensions {prediction.shape} is not the same as the target: {actuals.shape}"
        raise ValueError(msg)


def _compare_x_dims(featureMatrix, targetVector):
    if featureMatrix.shape[0] != targetVector.shape[0]:
        msg = f"Feature inpuit dimensions {featureMatrix.shape} does not match with target: {targetVector.shape}"
        raise ValueError(msg)


# def hyperparameter_search_BC(
#     trial,
#     hyperparameter_selector_function,
#     X,
#     y,
#     validation_data=None,
#     metric_function="accuracy",
#     score_on_proba=False,
#     metric_args=dict(),
# ):
#     """
#     Hyperparameter optmization function for BinaryClassifier
#     Parameters
#     ----------
#     trial: trial object from optuna
#     hyperparameter_selector_function: callable
#         Returns str, dict
#     X: pd.DataFrame
#     y: pd.Series or np.ndarray
#     validation data: tuple
#         Validation data in form of (X, y)
#     metric: callable
#         Receives two arguments(y_true and y_pred), should return a number
#         if `validation_data` is provided, returns calulation on validation data
#     score_on_proba: bool:
#         Computes metric on `y_true` VS `y_proba` instead of `y_true` vs `y_pred`


#     Returns
#     -------
#     metric: float
#         model score to minimize/maximize
#     """
#     # Hyperparameters
#     h_algorithm, model_hyperparameters = hyperparameter_selector_function(trial)

#     # Build model based on hyperparameters selected
#     binary_classifier = BinaryClassifierModel(algorithm=h_algorithm, model_args=model_hyperparameters)
#     binary_classifier.fit(X, y)

#     # Evaluate Model
#     if validation_data:
#         X_val, y_val = validation_data
#     else:
#         X_val, y_val = X, y
#         # TODO: Verify validation data using _validate_input_XY_pair(X_val,y_val)

#     if score_on_proba:
#         model_score = metric_function(y_true, y_proba, **metric_args)
#     else:
#         model_score = metric_function(y_true, y_pred, **metric_args)

#     return model_score

#     y_proba = binary_classifier.predict_proba(X_val)
#     y_pred = binary_classifier.predict(X_val)
#     y_true = y_val

#     # Determine metric based on the input
#     if type(metric) == str:
#         # use `binary_metrics` dictionary mapping
#         _validate_metric_name(metric, BINARY_METRICS_MAPPING)
#         metric_function = BINARY_METRICS_MAPPING.get(metric, None)

#         if metric == "roc_auc":
#             score_on_proba = True
#     else:
#         # uses callable
#         metric_function = metric

#     # Whether use the probabilities or the actual output
#     if score_on_proba:
#         model_score = metric_function(y_true, y_proba, **metric_args)
#     else:
#         model_score = metric_function(y_true, y_pred, **metric_args)

#     return model_score


class BinaryClassifierOptimizer(HyperParameterOptimizer):
    # Determine metric based on the input
    # if type(metric) == str:
    #     # use `binary_metrics` dictionary mapping
    #     _validate_metric_name(metric, BINARY_METRICS_MAPPING)
    #     metric_function = BINARY_METRICS_MAPPING.get(metric, None)

    #     if metric == "roc_auc":
    #         self.score_on_proba = True
    # else:
    #     # uses callable
    #     metric_function = metric
    pass


class RegressorOptimizer(HyperParameterOptimizer):
    pass


class ClusteringOptimizer(HyperParameterOptimizer):
    pass


def _validate_algorithm_hyperparameters(algorithm, hyperparameters):
    if type(hyperparameters) != dict:
        msg = f"Hyperparameters selected should be python dictionary. Selected : {str(hyperparameters)}"
        raise ValueError(msg)


def _validate_metric_name(metric, mapping):
    if metric not in set(mapping.keys()):
        msg = f"Metric name `{metric}` is not in {list(mapping.keys())}"
        raise ValueError(msg)


def _validate_input_XY_pair(X, y):
    pass
    # verify length


def run_hyperparameter_optmization(
    hyperparameter_selector_function,
    X,
    y,
    validation_data=None,
    metric="accuracy",
    score_on_proba=False,
    metric_args=dict(),
    n_trials=10,
    direction="maximize",
):
    # Create Optuna Study, specify if minimize or maximize the metric
    study = optuna.create_study(direction=direction)

    # Wrap the objective inside a lambda and call objective inside it
    objective_function = lambda trial: hyperparameter_search_BC(
        trial,
        hyperparameter_selector_function=hyperparameter_selector_function,
        X=X,
        y=y,
        validation_data=validation_data,
        metric=metric,
        score_on_proba=score_on_proba,
        metric_args=metric_args,
    )

    # Run objective function multiple times to search for the best trial
    study.optimize(objective_function, n_trials=n_trials)

    return study
