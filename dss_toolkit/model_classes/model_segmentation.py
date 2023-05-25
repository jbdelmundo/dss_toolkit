import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score, silhouette_samples

import joblib

class CustomEstimator(BaseEstimator):

    def __init__(
        self, 
        estimator = KMeans()
    ):
        """
        A Custom BaseEstimator that can switch between clustering models.
        :param estimator: sklearn object - The clustering model
        """ 

        self.estimator = estimator

    """
    Compute clustering.
    """

    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self

    """
    Predict the closest cluster each sample in X belongs to.
    """

    def predict(self, X, y=None):
        return self.estimator.predict(X)

    """
    Compute clustering and transform X to cluster-distance space.    
    """

    def fit_transform(self, X):
        return self.estimator.fit_transform(X)

        
    """
    Opposite of the value of X on the K-means objective.
    """

    def score(self, X, y):
        return self.estimator.score(X, y)
    
    
class ModelSegmentation:

    def __init__(
        self, 
        model_name : str = None,
        base_name : str = None,
        partition : int = None,
        data : pd.DataFrame = None,
        features : list = None,
        label_map : dict = None,
        model_artifact = None,
       
    ):
        """
        A Custom BaseEstimator that can switch between clustering models.
        :param data: pandas DataFrame object containing the data
        :param features: python list object containing the list of features used in clustering model
        :param label_map: python dictionary object containing the mapping of cluster index labels to cluster labels 
        :param model_artifact: python object containing the model to be used
        
        """ 


        """
        Raw Data Input
        """
        
        self.model_name = model_name
        self.base_name  = base_name
        self.partition = partition

        self.data = data

        """
        List of features used in clustering model
        """
        self.features = features

        """
        Mapping of Individual Cluster Label to Cluster Name
        
        Example: {0: 'Cluster Name 1', 1 : 'Cluster Name 2', 2: 'Cluster Name 3'}
        """
        self.label_map = model_artifact["cluster_names"]

        """
        Location of the pickle file of dimension reduction model used
        """
        self.model_artifact  = model_artifact

        """
        Cluster Assignment for each records
        """
        self.predictions = None

        """
        Silhouette Scores for each records
        """
        self.silhouette_scores = None

        """
        Average Silhouette Scores for each Clusters
        """
        self.individual_clusters_scores = None

        """
        Final Output Data Frame for Model Segmentation  
        """
        self.results = None

        
    def set_cluster_assignments(self):

        processed_data = self.data[self.features]

        self.predictions = self.model_artifact['estimator'].predict(processed_data)

    def get_clusters(self):

        self.set_cluster_assignments()
        return self.predictions

    def get_silhouette_scores(self):

        processed_data = self.data[self.features]

        if self.model_artifact['estimator'][:-1] is not None:
            processed_data = self.model_artifact['estimator'][:-1].transform(processed_data)

        self.silhouette_scores = silhouette_samples(processed_data, self.predictions)

        return self.silhouette_scores


    def get_customer_assignments(self):

        self.results = pd.DataFrame({'mastercif': self.data.mastercif, 
                        'cluster': self.predictions,
                        'cluster_name': None
                       })

        self.results.cluster_name = self.results.cluster.map(self.label_map)

    def get_individual_clusters_scores(self):

        if self.predictions is not None:
            silhouette_scores = self.get_silhouette_scores()
            cluster_numbers = list(np.unique(self.predictions))

            clusters_scores = []
            for cluster_label in cluster_numbers:
                clusters_scores.append(silhouette_scores[self.predictions == cluster_label].mean())

            self.individual_clusters_scores = dict(zip(cluster_numbers, clusters_scores))

            return self.individual_clusters_scores
        
        
    def get_performance(self):
        
        main_fields = ["model","base", "silhouette_score"]
        
        cluster_fields = ["cluster_{0}_sil_score".format(cluster) for cluster in range(1,11)]
        model_cluster_fields = ["cluster_{0}_sil_score".format(cluster) for cluster in self.individual_clusters_scores]

        columns = main_fields + cluster_fields + ["yearmo"]
        
        keys = main_fields + model_cluster_fields + ["yearmo"]
        
        silhouette_score = self.silhouette_scores.mean()
        individual_silhouette_scores = list(map(lambda key: self.individual_clusters_scores[key], self.individual_clusters_scores.keys()))
        
        values = [self.model_name, self.base_name, silhouette_score] + individual_silhouette_scores  + [self.partition]
        
        model_performance = dict(zip(keys, values))
        
        performance = pd.DataFrame(columns = columns)
        
        results = pd.DataFrame([model_performance])
        performance = pd.concat([performance, results], ignore_index=True)
        
        return performance






        
