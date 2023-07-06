import xgboost as xgb

from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MiniBatchKMeans

CLASSIFICATION = {
    "xgboost": xgb.XGBClassifier,
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "k_neighbors": KNeighborsClassifier,
    "lda": LinearDiscriminantAnalysis,
    "qda": QuadraticDiscriminantAnalysis,
}

REGRESSION = {
    "xgboost": xgb.XGBRegressor,
    "random_forest": RandomForestRegressor,
    "linear_regression": LinearRegression,
    "lasso_regression": Lasso,
}

CLUSTERING = {
    "kmeans": KMeans,
    "minibatch_kmeans": MiniBatchKMeans,
    "dbscan": DBSCAN,
    "agglomerative": AgglomerativeClustering,
}
