from sklearn.cluster import KMeans


def train_kmeans(train_data, train_labels, test_data=None, test_labels=None, *args, **kwargs):

    n_clusters = kwargs.get("n_clusters", 5)
    points_indices = kwargs.get("kmeans_indices", None)  # Select only some points to cluster
    kmeans_model = KMeans(n_clusters, random_state=5)

    if points_indices is None:
        kmeans_model.fit(train_data)  #
    else:
        kmeans_model.fit(train_data[points_indices])
    return kmeans_model


def predict_kmeans_transform(model, X, y=None):
    dist_to_centroids = model.transform(X)  # distance to k centroids
    return dist_to_centroids  # embedding


def predict_kmeans_cluster(model, X, y=None):
    return model.predict(X)  # Cluster Assignment
