from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score


def generate_clustering_report(X, cluster_ids):
    sample_size = None

    if X.shape[0] > 10000:
        sample_size = 10000  # Use Sampling if number of points > 10k

    return {
        "silhouette_score": silhouette_score(X, cluster_ids, sample_size=sample_size),
        "davies_bouldin_score": davies_bouldin_score(X, cluster_ids),
    }
