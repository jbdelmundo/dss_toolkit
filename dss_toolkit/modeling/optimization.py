import numpy as np
import pandas as pd
import time
import gc
from dss_toolkit.modeling.base import train_test_oot_pipeline


def create_hyperparameter_list(search_space):
    """
    Creates a dictionary of parameters.
    Sample input
    ``
    search_space = {
        "xgb_objective": ["binary:logistic", "rank:pairwise"],
        "xgb_eta": np.arange(0.05, 0.5, 0.05),
    }
    ``
    Sample output
    [
        {'xgb_objective': 'binary:logistic', 'xgb_eta': 0.05},
        {'xgb_objective': 'binary:logistic', 'xgb_eta': 0.1},
        {'xgb_objective': 'binary:logistic', 'xgb_eta': 0.15},
        {'xgb_objective': 'binary:logistic', 'xgb_eta': 0.2},
        {'xgb_objective': 'binary:logistic', 'xgb_eta': 0.25},
        {'xgb_objective': 'binary:logistic', 'xgb_eta': 0.3},
        {'xgb_objective': 'binary:logistic', 'xgb_eta': 0.35},
        {'xgb_objective': 'binary:logistic', 'xgb_eta': 0.4},
        {'xgb_objective': 'binary:logistic', 'xgb_eta': 0.45},
        {'xgb_objective': 'rank:pairwise', 'xgb_eta': 0.05},
        {'xgb_objective': 'rank:pairwise', 'xgb_eta': 0.1},
        {'xgb_objective': 'rank:pairwise', 'xgb_eta': 0.15},
        {'xgb_objective': 'rank:pairwise', 'xgb_eta': 0.2},
        {'xgb_objective': 'rank:pairwise', 'xgb_eta': 0.25},
        {'xgb_objective': 'rank:pairwise', 'xgb_eta': 0.3},
        {'xgb_objective': 'rank:pairwise', 'xgb_eta': 0.35},
        {'xgb_objective': 'rank:pairwise', 'xgb_eta': 0.4},
        {'xgb_objective': 'rank:pairwise', 'xgb_eta': 0.45}
    ]

    Parameter
    ---------
    search_space: dict
        Each key is a the hyperparameter and value is a list of values for the hyperparameter
    """

    hyperparams_list_df = pd.DataFrame([{"merge_key": 1}])
    for k, v in search_space.items():
        df = pd.DataFrame({k: v})
        df = df.fillna(np.nan).replace([np.nan], [None])
        df["merge_key"] = 1

        hyperparams_list_df = hyperparams_list_df.merge(df, on="merge_key")

    hyperparams_list_df.drop(columns="merge_key", inplace=True)
    hyperparams_list = hyperparams_list_df.to_dict(orient="records")
    return hyperparams_list


def grid_search(
    X_train,
    y_train,
    X_test,
    y_test,
    X_oot,
    y_oot,
    data_preprocessor_function=None,
    train_model_function=None,
    predict_model_function=None,
    categorical_columns=None,
    numeric_columns=None,
    hyperparams_list=[],
):

    all_reports = []
    start_time = time.process_time()

    # Start hyperparameter search
    for ix, hyperparameter in enumerate(hyperparams_list):
        print(f"Evaluating {ix+1} of {len(hyperparams_list)} hyperparameters")
        try:
            # Evaluate a single set of hyperparameter configuration
            reports, runs = train_test_oot_pipeline(
                X_train,
                y_train,
                X_test,
                y_test,
                X_oot,
                y_oot,
                data_preprocessor_function=data_preprocessor_function,
                train_model_function=train_model_function,
                predict_model_function=predict_model_function,
                categorical_columns=categorical_columns,
                numeric_columns=numeric_columns,
                **hyperparameter,
            )
            del runs
            gc.collect()
            reports["run_id"] = ix
            all_reports.append(reports)
        except Exception as e:
            print(f"Error at run {ix} {str(e)}")

    end_time = time.process_time()
    duration = end_time - start_time
    print("Duration:", duration)

    # Setup hyperparameter dataframe
    hyperparams_df = pd.DataFrame(hyperparams_list)
    hyperparams_df["run_id"] = list(range(len(hyperparams_df)))

    # Combine all reports into a single dataframe
    search_reports = pd.concat(all_reports, axis=0)
    search_reports.reset_index(inplace=True)
    search_reports.rename(columns={"index": "run_data"}, inplace=True)

    # Merge model performance report with hyperparameters
    return search_reports.merge(hyperparams_df, on="run_id")
    
