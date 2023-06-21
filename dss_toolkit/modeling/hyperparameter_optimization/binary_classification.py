import sklearn
import optuna
from dss_toolkit.modeling.base import BinaryClassifier

BINARY_METRICS_MAPPING = {
    "accuracy": sklearn.metrics.accuracy_score,
    "precision": sklearn.metrics.precision_score,
    "recall": sklearn.metrics.recall_score,
    "f1": sklearn.metrics.f1_score,
    "roc_auc": sklearn.metrics.roc_auc_score,
}


def hyperparameter_search_BC(
    trial,
    hyperparameter_selector_function,
    X,
    y,
    validation_data=None,
    metric="accuracy",
    score_on_proba=False,
    metric_args=dict(),
):
    """
    Hyperparameter optmization function for BinaryClassifier
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

    # Build model based on hyperparameters selected
    binary_classifier = BinaryClassifier(algorithm=h_algorithm, model_args=model_hyperparameters)
    binary_classifier.fit(X, y)

    # Generate predictions
    if validation_data:
        X_val, y_val = validation_data
        # TODO: Verify validation data using _validate_input_XY_pair(X_val,y_val)

        y_proba = binary_classifier.predict_proba(X_val)
        y_pred = binary_classifier.predict(X_val)
        y_true = y_val
    else:
        # Use Training data if no validation data is provided
        y_proba = binary_classifier.predict_proba(X)
        y_pred = binary_classifier.predict(X)
        y_true = y

    # Determine metric based on the input
    if type(metric) == str:
        # use `binary_metrics` dictionary mapping
        _validate_metric_name(metric, BINARY_METRICS_MAPPING)
        metric_function = BINARY_METRICS_MAPPING.get(metric, None)

        if metric == "roc_auc":
            score_on_proba = True
    else:
        # uses callable
        metric_function = metric

    # Whether use the probabilities or the actual output
    if score_on_proba:
        model_score = metric_function(y_true, y_proba, **metric_args)
    else:
        model_score = metric_function(y_true, y_pred, **metric_args)

    return model_score


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
    )

    # Run objective function multiple times to search for the best trial
    study.optimize(objective_function, n_trials=n_trials)

    return study
