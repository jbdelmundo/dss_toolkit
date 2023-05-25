import numpy as np
import pandas as pd

# from dss_toolkit.modeling.smote import smote_py
from sklearn.preprocessing import RobustScaler


# def preprocess_onehot_then_smote(X, y, *args, **kwargs):
#     required_columns = kwargs.get("required_columns", None)
#     is_test_data = kwargs.get("is_test_dasdasata", False)

#     X_dummies = _preprocess_onehot(X, y, required_columns=required_columns)
#     preprocessor_data = None  # Some preprocessing could have a stored model (like PCA model)

#     if not is_test_data:
#         X_smote, y_smote = _preprocess_smote(X_dummies, y, **kwargs)
#         return (X_smote, y_smote), preprocessor_data
#     else:
#         return (X_dummies, y), preprocessor_data


def _preprocess_onehot(X, y=None, **kwargs):
    required_columns = kwargs.get("required_columns", None)

    X_dummies = pd.get_dummies(X)

    if required_columns is not None:
        # set columns
        if len(required_columns) == 0:
            raise Exception("Empty list for argument `required_columns`")

        # Add missing columns
        for req_col in required_columns:
            if req_col not in X_dummies.columns:
                X_dummies[req_col] = 0

        # Remove extra columns
        extra_cols = [c for c in X_dummies.columns if c not in required_columns]

        X_dummies = X_dummies.drop(columns=extra_cols)
        X_dummies = X_dummies[required_columns]

    return X_dummies


# def _preprocess_smote(X, y, *args, **kwargs):

#     smote_oversample = kwargs.get("smote_oversample", None)
#     smote_undersample = kwargs.get("smote_undersample", None)

#     # Apply SMOTE
#     X_smote, y_smote = smote_py(
#         X,
#         y,
#         oversample=smote_oversample,  # new proportion(minority/majority) after oversampling
#         undersample=smote_undersample,  # new proportion(minority/majority) after undersampling
#     )
#     return X_smote, y_smote


def preprocess_scale_onehot(X, y, *args, **kwargs):

    preprocessor_data = kwargs.get("preprocessor_data", dict())
    is_test_data = kwargs.get("is_test_data", False)

    categorical_columns = kwargs.get("categorical_columns", [])
    numeric_columns = kwargs.get("numeric_columns", [])

    if is_test_data:
        X_dummies = _preprocess_onehot(
            X[categorical_columns], y, required_columns=preprocessor_data.get("dummies_cols", None),
        )
        rs = preprocessor_data.get("scaler", None)
    else:
        X_dummies = _preprocess_onehot(X[categorical_columns], y, **kwargs)
        rs = RobustScaler(with_centering=False)

    X = X.copy()
    X.loc[:, numeric_columns] = rs.fit_transform(X.loc[:, numeric_columns])
    X_preprocessed = pd.concat([X[numeric_columns], X_dummies], axis=1)

    #     print("Preprocess", X_preprocessed.shape)
    return (X_preprocessed, y), {"scaler": rs, "dummies_cols": X_dummies.columns}


def preprocess_scale_onehot_drop_correlated(X, y, *args, **kwargs):

    correlation_threshold = kwargs.get("correlation_threshold", 0.5)
    is_test_data = kwargs.get("is_test_data", False)
    preprocessor_data = kwargs.get("preprocessor_data", dict())
    to_drop = kwargs.get("columns_to_drop", None)

    (X_preprocessed, y), preprocessordata = preprocess_scale_onehot(X, y, **kwargs)

    # Columns to drop are specified
    if to_drop is not None:
        pass

    elif not is_test_data:
        # Create correlation matrix
        corr_matrix = X_preprocessed.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # Find features with correlation greater than `correlation_threshold`
        to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
    else:
        # Get from preprocessor data
        to_drop = preprocessor_data["correlated_to_drop"]

    # Drop features
    X_preprocessed.drop(to_drop, axis=1, inplace=True)
    preprocessordata["correlated_to_drop"] = to_drop

    return (X_preprocessed, y), preprocessordata

