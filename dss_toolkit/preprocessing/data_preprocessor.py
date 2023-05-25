from dss_toolkit.preprocessing.feature_selection import (
    find_VIF,
    appy_boruta,
    get_low_variance_columns,
    find_column_correlations,
)
from sklearn.model_selection import train_test_split



# Feature Selection functions
feature_selection_functions = {
    "feature_selection_vif": find_VIF,
    "feature_selection_boruta": appy_boruta,
    "feature_selection_low_variance": get_low_variance_columns,
    "feature_selection_column_correlation": find_column_correlations,
}




class DataProcessor:
    def __init__(self):
        self.data_df = None
        self.train_df = None
        self.test_df = None
        self.val_df = None

        self.categorical_columns = []
        self.numeric_columns = []
        self.target_column = None

        # List of transformers/encoders for pipeline
        self.pipe = []

        self.transformer_artifacts = dict()

    def set_data(self, data_df):
        self.data_df = data_df

    def set_data_types(self, categorical_columns, numeric_columns, target_column):
        if self.data_df is None:
            raise Exception("Data not set. Use set_data()")
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.target_column = target_column

    # =================== Data Split  ===================
    def set_validation_data(self, validation_df):
        self.val_df = validation_df

    def split_data(self, method="train_test_split", custom_function=None, **kwargs):
        if self.data_df is None:
            raise Exception("Data not set. Use set_data()")

        if method == "train_test_split":

            (train_df, test_df) = train_test_split(
                self.data_df,
                train_size=kwargs.get("train_size", None),
                test_size=kwargs.get("test_size", None),
                random_state=kwargs.get("random_state", None),
                shuffle=kwargs.get("shuffle", True),
                stratify=kwargs.get("stratify", None),
            )

            self.train_df = train_df
            self.test_df = test_df

        elif method == "kfold":
            print("KFold Split not yet implemented")

        elif method == "time_series":
            print("Time Series Split not yet implemented")

        elif method == "custom":
            return custom_function(self.data_df, **kwargs)
        else:
            raise Exception(f"Unknown `method` parameter:{method} ")

    #  =================== Data Preprocessing  ===================

    def apply_preprocessor(self, data_df, preprocessor, **kwargs):

        if preprocessor in cleaning_functions.keys():
            clean_function = cleaning_functions[preprocessor]
            cleaned_df = clean_function(data_df, **kwargs)
            return cleaned_df

        elif preprocessor in feature_selection_functions.keys():
            feature_selector_function = feature_selection_functions[preprocessor]
            columns = feature_selector_function(data_df, **kwargs)
            return data_df[columns]
        else:
            raise Exception(f"Unknown `preprocessor` value:{preprocessor} ")

    def add_transformer(self, preprocessor, **kwargs):

        if preprocessor in feature_transformation_classes.keys():
            TransformerClass = feature_transformation_classes[preprocessor]
            self.pipe.append((preprocessor, TransformerClass(**kwargs)))
        else:
            TransformerClass = preprocessor
            self.pipe.append(("custom_transformer", TransformerClass(**kwargs)))

    def create_pipeline(self):
        if self.pipes:
            return Pipeline(self.pipes)
        else:
            raise Exception("Empty transformers. Add `add_transformer`")

