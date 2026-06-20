import numpy as np
from scipy.spatial.distance import pdist

def compute_sum_distances_between_centroids_and_global_centroid(X, labels):

    # Get the number of anti-clusters
    n_anticlusters = len(np.unique(labels))

    # Compute the count of each label (for averaging)
    counts = np.bincount(labels, minlength=n_anticlusters).reshape(-1, 1)

    # Sum features per cluster
    sums = np.zeros((n_anticlusters, X.shape[1]))
    for i in range(X.shape[1]):
        sums[:, i] = np.bincount(labels, weights=X[:, i], minlength=n_anticlusters)

    # Divide sum by count to get means
    cluster_centers = sums / counts

    # Get global centroid
    global_centroid = X.mean(axis=0)

    # Compute k-means clustering costs
    return ((cluster_centers - global_centroid)**2).sum(axis=1).sum()


def compute_sum_distances_to_centroids(X, labels):

    # Get the number of anti-clusters
    n_anticlusters = len(np.unique(labels))

    # Compute the count of each label (for averaging)
    counts = np.bincount(labels, minlength=n_anticlusters).reshape(-1, 1)

    # Sum features per cluster
    sums = np.zeros((n_anticlusters, X.shape[1]))
    for i in range(X.shape[1]):
        sums[:, i] = np.bincount(labels, weights=X[:, i], minlength=n_anticlusters)

    # Divide sum by count to get means
    cluster_centers = sums / counts

    # Compute k-means clustering costs
    return ((X - cluster_centers[labels, :])**2).sum(axis=1).sum()

def compute_sum_pairwise_distances(X, labels):
        
    diversity = 0.0
    for label in np.unique(labels):
        objects = X[labels == label]
        if len(objects) >= 2:
            dists = pdist(objects, metric='sqeuclidean')  
            diversity += np.sum(dists)

    return diversity
