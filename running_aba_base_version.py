# © 2026, University of Bern, Group for Business Analytics, Operations Research and Quantitative Methods,
# Philipp Baumann

import time
import pandas as pd
from toolbox import compute_sum_pairwise_distances, compute_sum_distances_to_centroids
from aba import run_aba

# Load data
X = pd.read_csv('datasets/npi.csv', header=None).values.astype(float)

# Set number of anticlusters
n_anticlusters = 5

# Start stopwatch
tic = time.perf_counter()

# Run approach
labels = run_aba(X, n_anticlusters)

# Get runtime
runtime = time.perf_counter() - tic

# Compute diversity
diversity = compute_sum_pairwise_distances(X, labels)

# Print results
print(f'Runtime: {runtime:.2f} seconds')
print(f'Sum of pairwise distances: {compute_sum_pairwise_distances(X, labels):.2f}')
print(f'Sum of distances to centroids: {compute_sum_distances_to_centroids(X, labels):.2f}')