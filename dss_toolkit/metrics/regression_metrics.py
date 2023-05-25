from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_absolute_error,
    max_error,
    explained_variance_score,
)


def generate_regression_report(targets, preds):
    reg_metrics = {
        "len": len(targets),
        "r2": r2_score(targets, preds),
        "mape": mean_absolute_percentage_error(targets, preds),
        "mse": mean_squared_error(targets, preds),
        "rmse": mean_squared_error(targets, preds, squared=False),
        "mae": mean_absolute_error(targets, preds),
        #         "msle": mean_squared_log_error(targets, preds), # Check compatible sklearn version
        #         "rmsle": mean_squared_log_error(targets, preds,squared=False),
        "max_error": max_error(targets, preds),
        "explained_variance_score": explained_variance_score(targets, preds),
    }
    return reg_metrics
    