import pandas as pd

from dss_toolkit.modeling.model_metrics_old import (
    generate_classification_report,
    decile_performance,
    generate_clustering_report,
    generate_regression_report,
)
from dss_toolkit.modeling.algo.cluster import train_kmeans, predict_kmeans_cluster, predict_kmeans_transform


import traceback


def train_pipeline(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    data_preprocessor_function=None,
    train_model_function=None,
    predict_model_function=None,
    cluster_assignment_function=None,
    learning="classification",  # or "regression"
    **kwargs,
):
    """
    Standard Pipeline for Training and Evaluating model
    1. Preprocess Data (Training and Test Data)
    2. Train Model
    3. Generate Prediction (Training and Test)

    Returns
    ------
    Dictionary containing:
    - Model performance
    - Decile Report
    - Predictions
    - Trained model
    - Data Preprocessor information
    - Model Variables (Train columns)

    """

    # Data Preprocessing- Train Data
    if data_preprocessor_function is not None:
        (train_data, train_labels), preprocessor_data = data_preprocessor_function(X_train, y_train, **kwargs)
    else:
        (train_data, train_labels) = X_train, y_train

    # Data Preprocessing- Validation Data (for early stoping is supported by `train_model_function`)
    if X_val is not None and y_val is not None and data_preprocessor_function is not None:
        (val_data, val_labels), _ = data_preprocessor_function(
            X_val,
            y_val,
            is_test_data=True,  # Special process for test data
            preprocessor_data=preprocessor_data,  # Special variables from training data (like transformers)
            required_columns=train_data.columns,
            **kwargs,
        )
    else:
        (val_data, val_labels) = None, None

    # Model Training and prediction
    model = train_model_function(train_data, train_labels, val_data, val_labels, **kwargs)
    train_proba = predict_model_function(model, train_data, train_labels)

    # Metrics
    if learning == "classification":
        train_report = generate_classification_report(train_labels, train_proba)

    elif learning == "regression":
        train_report = generate_regression_report(train_labels, train_proba)

    else:
        train_report = None  # To add support for unsupervised

    train_decile = decile_performance(train_labels, train_proba)

    # For clustering evaluation (If model uses clustering for "supervised" approach)
    if cluster_assignment_function is not None:
        try:
            cluster_ids = cluster_assignment_function(model, train_data)  # Assign clusters
            cluster_report = generate_clustering_report(train_data, cluster_ids)
        except Exception as e:
            cluster_report = str(e)
    else:
        cluster_report = None

    return {
        "reports": train_report,
        "cluster_report": cluster_report,
        "results": (train_labels, train_proba),
        "decile_performance": train_decile,
        "model": model,
        "preprocessor_data": preprocessor_data,
        "train_columns": train_data.columns,
    }


def inference_pipeline(
    X,
    y,
    model,
    data_preprocessor_function=None,
    predict_model_function=None,
    cluster_assignment_function=None,
    learning="classification",  # or "regression"
    **kwargs,
):
    """
    Standard Pipeline for Model Infenrence (Prediction for OOT/Prod)
    1. Data Preprocessing
    2. Prediction
    3. Model Performance Reports (Accuracy, AUC-ROC, etc)
    5. Model Decile Report (conversion rate)
    
    Returns
    ------
    Dictionary containing
    - Model performance
    - Decile Report
    - Predictions
    - Data Preprocessor Used
    - Model Used
    """

    preprocessor_data = kwargs.get("preprocessor_data", None)

    # Data Preprocessing
    (features, labels), _ = data_preprocessor_function(
        X, y, is_test_data=True, **kwargs,  # Special process for test data
    )

    # Model Evaluation
    proba = predict_model_function(model, features, labels)

    # Compute metrics if labels are provided (Test data and OOT)
    if y is not None:
        if learning == "classification":
            report = generate_classification_report(labels, proba)
        elif learning == "regression":
            report = generate_regression_report(labels, proba)
        else:
            # To Add unsupervised reporting
            report = None

        decile_perf = decile_performance(labels, proba)
    else:
        report = None
        decile_perf = None

    # For clustering evaluation (only for those using clustering approach)
    if cluster_assignment_function is not None:
        try:
            cluster_ids = cluster_assignment_function(model, features)  # Assign clusters
            unique_clusters = pd.Series(cluster_ids).unique()
            if unique_clusters.shape[0] > 0:
                cluster_performance = generate_clustering_report(features, cluster_ids)
            else:
                cluster_performance = dict()
        except Exception as e:
            tb = traceback.format_exc()
            cluster_performance = "Error in assigning clusters " + str(e) + "\n" + str(tb)
    else:
        cluster_performance = None

    return {
        "reports": report,
        "cluster_report": cluster_performance,
        "results": (labels, proba),
        "decile_performance": decile_perf,
        "model": model,
        "preprocessor_data": preprocessor_data,
    }


def train_test_oot_pipeline(
    X_train,
    y_train,
    X_test,
    y_test,
    X_oot,
    y_oot,
    data_preprocessor_function,
    train_model_function,
    predict_model_function,
    categorical_columns,
    numeric_columns,
    learning="regression",
    return_runs=True,
    **kwargs,
):
    """
    Combined Pipeline for doing train-test-oot process.
    Internally calls `train_pipeline()` for training data, `inference_pipeline()` for test and OOT data

    Returns
    -------
    Reports : Metrics for the ML
    Pipline Results:
        If `return_runs` set to True
        Returns individual runs per pipeline (train, test and OOT)
    """

    run_base = train_pipeline(
        X_train,
        y_train,
        X_test,
        y_test,
        data_preprocessor_function=data_preprocessor_function,
        train_model_function=train_model_function,
        predict_model_function=predict_model_function,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        learning=learning,
        **kwargs,
    )

    run_test = inference_pipeline(
        X_test,
        y_test,
        model=run_base["model"],
        data_preprocessor_function=data_preprocessor_function,
        train_model_function=train_model_function,
        predict_model_function=predict_model_function,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        preprocessor_data=run_base["preprocessor_data"],
        learning=learning,
        **kwargs,
    )

    run_oot = inference_pipeline(
        X_oot,
        y_oot,
        model=run_base["model"],
        data_preprocessor_function=data_preprocessor_function,
        train_model_function=train_model_function,
        predict_model_function=predict_model_function,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        preprocessor_data=run_base["preprocessor_data"],
        learning=learning,
        **kwargs,
    )

    train_report = run_base["reports"]
    test_report = run_test["reports"]
    oot_report = run_oot["reports"]

    reports_df = pd.DataFrame([train_report, test_report, oot_report])
    reports_df.index = ["Train", "Test", "OOT"]

    if return_runs:
        return reports_df, {"train": run_base, "test": run_test, "oot": run_oot}
    return reports_df

