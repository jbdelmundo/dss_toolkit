from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score


def generate_report(X, cluster_ids, sample_points=10000):
    sample_size = None

    if X.shape[0] > sample_points:
        sample_size = sample_points  # Use Sampling if number of points > 10k, silhouette_score score runs longer on larger dataset

    return {
        "silhouette_score": silhouette_score(X, cluster_ids, sample_size=sample_size),
        "davies_bouldin_score": davies_bouldin_score(X, cluster_ids),
    }
