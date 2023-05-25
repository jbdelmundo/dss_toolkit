import xgboost as xgb


def train_xgboost_classifier(train_data, train_labels, test_data, test_labels, **kwargs):

    xgb_params = {
        "booster": kwargs.get("xgb_booster", "gbtree"),
        "objective": kwargs.get("xgb_objective", "binary:logistic"),
        "eta": kwargs.get("xgb_eta", 0.3),
        "gamma": kwargs.get("xgb_gamma", 0.1),
        "max_depth": kwargs.get("xgb_max_depth", 9),
        "min_child_weight": kwargs.get("xgb_min_child_weight", 1),
        "subsample": kwargs.get("xgb_subsample", 1),
        "colsample_bytree": kwargs.get("xgb_colsample_bytree", 0.75),
        "nthread": kwargs.get("xgb_nthread", -1),
        "eval_metric": kwargs.get("xgb_eval_metric", "auc"),  # negative log loss
        "seed": 23,
    }

    num_rounds = kwargs.get("xgb_num_rounds", 20)
    early_stopping_rounds = kwargs.get("xgb_early_stopping_rounds", 10)

    dtrain = xgb.DMatrix(train_data, label=train_labels, enable_categorical=True)
    dtest = xgb.DMatrix(test_data, label=test_labels, enable_categorical=True)
    watchlist = [(dtrain, "train"), (dtest, "test")]

    xgb0 = xgb.train(
        xgb_params,
        dtrain,
        num_rounds,
        watchlist,
        early_stopping_rounds=early_stopping_rounds,
        maximize=kwargs.get("xgb_maximize", True),
        verbose_eval=0,
    )

    return xgb0


def predict_xgboost(model, X, y):
    xgb_data = xgb.DMatrix(X, label=y, enable_categorical=True)
    proba = model.predict(xgb_data)
    return proba
