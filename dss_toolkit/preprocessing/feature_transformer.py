from logging import raiseExceptions
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
    MinMaxScaler,
    QuantileTransformer,
    PowerTransformer,
    LabelBinarizer,
)
import pandas as pd
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.decomposition._base import _BasePCA

transformer_mappings = {
    "onehot_encoder": OneHotEncoder,
    "standard_scaler": StandardScaler,
    "robust_scaler": RobustScaler,
    "min_max_scaler": MinMaxScaler,
    "power_transformer": PowerTransformer,
    "label_binarizer": LabelBinarizer,
    "pca": PCA,
}


class DataFrameTransformer
    """
        Wrapper class for transforming Pandas data frames
    """

    def __init__(self, transformer=None):
        self.transformer=transformer

    def fit(self, X, y=None, **fit_params):
        return self.transformer.fit(X,y,**fit_params)

    def transform(self, X, y=None, **fit_params):
        return self.transformer.fit(X,y,**fit_params)

    def fit(self, X, y=None, **fit_params):
        return self.transformer.fit(X,y,**fit_params)

# class DataImputer:

#     # TO BE REPLACED BY Column transformer + Pipeline + SimpleImputer
#     def __init__(self, imputation_rules=dict()):
#         """
#         imputation_rules: dict
#             key: name of imputer (could be anything)
#             value: dict
#                 strategy : required ['mean','median','most_frequent','constant']
#                 fill_value : value to replace
#                 columns : list of column names affected
#         """
#         self.imputers = dict()
#         self.imputation_rules = imputation_rules

#     def fit(self, df):
#         for imputer_name, imputer_params in self.imputation_rules.items():
#             print(imputer_params)
#             df_columns = imputer_params.get("columns")
#             strategy = imputer_params.get("strategy")
#             fill_value = imputer_params.get("fill_value", None)

#             imp = SimpleImputer(strategy=strategy, fill_value=fill_value)
#             imp.fit(df[df_columns])
#             self.imputers[imputer_name] = imp

#     def transform(self, df):
#         for imputer_name, imputer_params in self.imputation_rules.items():
#             imp = self.imputers[imputer_name]
#             df_columns = imputer_params.get("columns")
#             df.loc[:, df_columns] = imp.transform(df[df_columns])
#         return df

#     def fit_transform(self, df):
#         self.fit(df)
#         return self.transform(df)


class DataTransformer:
    """
    Collection of classes with fit() and transform() used for preprocessing

    If sklearn.pipeline.Pipeline and sklearn.composer.ColumnTransformer
    can do a better job, pls halppp
    """

    def __init__(self, data=None):
        self.df = data
        self.transformers = list()  # list of dictionary

    def add_transformer(self, transformer, columns=None, include_columns=[], exclude_columns=[], **kwargs):
        """
        Add transformer object that supports `fit()`, `transform()` and `fit_transform()`
        Specify the columns to be affected

        Parameters
        -----------------
        transformer: string or Estimator
        columns: string ("all","categorical" or "numeric")
        include_columns: list of column names to be included
        exclude_columns: list of column names to be excluded

        """

        # Add support for custom transformer
        if type(transformer) == str:
            data_transformer = transformer_mappings[transformer](**kwargs)
        else:
            data_transformer = transformer

        # Check if transformer if it supports the required functions
        if (
            hasattr(data_transformer, "fit_transform")
            and hasattr(data_transformer, "fit")
            and hasattr(data_transformer, "transform")
        ):

            # Add to list of transformers
            self.transformers.append(
                {
                    "transformer": data_transformer,
                    "columns": columns,
                    "include_columns": include_columns,
                    "exclude_columns": exclude_columns,
                }
            )
        else:
            raise Exception("Transformer provided do not have fit(), transform() nad fit_transform()")

    def show_transformers(self):
        print("Showing all transformations in specific order:")
        for i, transformer_dict in enumerate(self.transformers):
            transformer = transformer_dict["transformer"]  # Class with fit and transform
            columns = transformer_dict["columns"]  # all, categorical or numeric
            include_columns = transformer_dict["include_columns"]  # Specific columns to include
            exclude_columns = transformer_dict["exclude_columns"]  # Specific columns to exclude)

            print(f"{i}. Transformer", transformer)
            print(f"\t Columns: {columns}")
            print("\t Included Columns: ", include_columns)
            print("\t Excluded Columns: ", exclude_columns, "\n")

    def apply_transformer(self, df, transformer_dict, method="fit_transform"):

        transformer = transformer_dict.get("transformer")  # Class with fit and transform
        columns = transformer_dict.get("columns", "all")  # all, categorical or numeric
        include_columns = transformer_dict.get("include_columns", [])  # Specific columns to include
        exclude_columns = transformer_dict.get("exclude_columns", [])  # Specific columns to exclude

        # Select columns to be affected
        if columns == "all":
            affected_columns = df.columns.tolist()
        elif columns == "numeric":
            affected_columns = df.select_dtypes(include="number").columns.tolist()  # Select all numeric columns
        elif columns == "categorical":
            affected_columns = df.select_dtypes(include="object").columns.tolist()  # Select all string columns
        elif columns is None:
            affected_columns = []
        else:
            raise Exception(
                f"Invalid value for parameter `columns`. Accepted values are 'all','numeric','categorical'. Found: {columns}. "
            )

        # Remove excluded columns and add explicit included columns
        affected_columns = [c for c in affected_columns if c not in exclude_columns] + include_columns
        print("Applying:", transformer, "\nAffected columns:", affected_columns, "\n")

        # Apply transformation
        if method == "transform":
            transformed = transformer.transform(df[affected_columns])
        else:
            transformed = transformer.fit_transform(df[affected_columns])

        # If number of columns is same as before transformation
        if transformed.shape[1] == len(affected_columns):
            df.loc[:, affected_columns] = transformed  # Replace column values

        else:
            # If One hot encoding is done, it changes the number of columns
            if isinstance(transformer, OneHotEncoder):
                # Drop original columns, merge encoded columns
                new_columns = transformer.get_feature_names(affected_columns)
                to_merge = pd.DataFrame(transformed, columns=new_columns)
                df.drop(columns=affected_columns, inplace=True)
                df = pd.concat([df, to_merge], axis=1)

            # if transformation is PCA
            elif isinstance(transformer, _BasePCA):
                # Replace the data frame with new columns
                new_columns = [f"pc_{i}" for i in range(transformer.n_components_)]
                df = pd.DataFrame(transformed, columns=new_columns)
            else:
                print(
                    f"Different dimensions detected after transformation. From {df[affected_columns].shape}  to  {transformed.shape}. Returning a numpy matrix instead"
                )
                df = transformed

        return df

    def fit(self, df, verbose=1):
        self.fit_transform(df, verbose)  # Call fit_transform without returning anything

    def fit_transform(self, df, verbose=1):

        if verbose:
            self.show_transformers()

        df = df.copy()
        # Iterate over the list of transformers and call fit_transform()
        for transformer_dict in self.transformers:
            df = self.apply_transformer(df, transformer_dict, method="fit_transform")
        return df

    def transform(self, df, verbose=1):
        if verbose:
            self.show_transformers()

        df = df.copy()
        # Iterate over the list of transformers and call transform()
        for transformer_dict in self.transformers:
            df = self.apply_transformer(df, transformer_dict, method="transform")

        return df

