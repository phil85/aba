# © 2026, University of Bern, Group for Business Analytics, Operations Research and Quantitative Methods,
# Philipp Baumann

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from joblib import Parallel, delayed


def process_cluster(cluster, labels, X, subclusters, categories):

    # Get cluster members
    idx = labels == cluster
    cluster_members = np.where(idx)[0]

    if categories is None:

        # Subdivide cluster into subclusters
        labels = run_aba(X[cluster_members], subclusters)

    else:

        # Subdivide cluster into subclusters
        labels = run_aba(X[cluster_members], subclusters, categories[cluster_members])


    return cluster, cluster_members, labels


def assign_objects(X, centers, batch, categories=None, category_upper_bounds=None, cluster_category_counts=None):

    # Compute model input
    distances = cdist(X[batch, :], centers, 'sqeuclidean')

    if categories is not None:

        # Get unique categories of objects in batch
        unique_categories = np.unique(categories)

        # Check for each cluster if category upper bound has been reached            
        for category in unique_categories:

            # Get clusters which are already full
            idx = cluster_category_counts[:, category] == category_upper_bounds[category]

            # Adjust distances to prevent a violation of the upper bound
            if idx.any():
                val = -distances.max().max()                
                distances[np.ix_(categories == category, idx)] = val

    # Solve assignment problem
    _, col_ind = linear_sum_assignment(distances, maximize=True)

    return col_ind


def get_batches(sorted_objects, n_clusters, categories=None, small_anticlusters_flag=False):

    if categories is None:

        # Adjustment that often improves the objective function value when k is large
        if small_anticlusters_flag:

            # If N is divisible by K
            if len(sorted_objects) % n_clusters == 0:

                # Get length of one sublist
                length_of_sublist = len(sorted_objects) // n_clusters

                # Split sorted objects into sublists
                sublists = np.array_split(sorted_objects, range(length_of_sublist, len(sorted_objects), length_of_sublist))

                # Stack sublists on top of each other
                stacked_sublists = np.vstack(sublists)

                # Create batches by taking columns of the stacked sublists
                batches = list(stacked_sublists.T)

            else:

                # Get minimum size of anticluster
                min_size_anticluster = len(sorted_objects) // n_clusters

                # Get maximum size of anticluster
                max_size_anticluster = min_size_anticluster + 1

                # Number of min-size anticlusters
                n_min_size_anticlusters = max_size_anticluster * n_clusters - len(sorted_objects)

                # Number of max-size anticlusters
                n_max_size_anticlusters = len(sorted_objects) - min_size_anticluster * n_clusters

                # Split sorted objects into min-size and max-size parts
                part1, part2 = np.split(sorted_objects, [n_min_size_anticlusters * min_size_anticluster])

                # Get number of short sublists
                short_sublists = np.array_split(part1, range(min_size_anticluster, len(part1), min_size_anticluster))

                # Get number of long sublists
                long_sublists = np.array_split(part2, range(max_size_anticluster, len(part2), max_size_anticluster))

                # Stack short sublists on top of each other
                stacked_short_sublists = np.vstack(short_sublists)

                # Stack long sublists on top of each other
                stacked_long_sublists = np.vstack(long_sublists)

                # Extract last column from stacked_long_sublists
                last_column = stacked_long_sublists[:, -1]

                # Stack short and long sublists (without last column) on top of each other
                combined_stacked_sublists = np.vstack((stacked_short_sublists, stacked_long_sublists[:, :-1]))

                # Create batches
                batches = list(combined_stacked_sublists.T) + [last_column]

        else: 

            # Split sorted objects into batches of size n_clusters
            batches = np.array_split(sorted_objects, range(n_clusters, len(sorted_objects), n_clusters))

        return batches            
    else:

        # Get unique categories
        unique_categories = np.unique(categories)

        # Initialize dictionaries
        complete_splits = {}
        incomplete_splits = {}

        for category in unique_categories:

            # Get all objects that belong to this category (according to sequence in sorted_objects)
            objects_of_category = sorted_objects[categories == category]

            # Split these objects into groups of size n_clusters
            splits = np.array_split(objects_of_category, range(n_clusters, len(objects_of_category), n_clusters))

            # Store complete and incomplete splits separately
            complete_splits[category] = splits[:-1]
            incomplete_splits[category] = splits[-1]

        # Get total number of complete splits
        n_complete_splits = sum(len(splits) for splits in complete_splits.values())

        # Initialize a dictionary that keeps track of the number of complete splits already assigned to batches
        split_counter = {cat: 0 for cat in unique_categories}

        # Create batches by looping through all complete splits (alternating between categories in each iteration)
        batches = []        
        while len(batches) < n_complete_splits:
            for cat in unique_categories:
                if split_counter[cat] < len(complete_splits[cat]):
                    batches.append(complete_splits[cat][split_counter[cat]])
                    split_counter[cat] += 1

        # Store incomplete batches in a list
        incomplete_batches = list(incomplete_splits.values())

        # Flatten batches to ensure that all batches except the last one have cardinality=n_clusters
        sorted_objects = np.concatenate(batches + incomplete_batches)
        splits = np.arange(n_clusters, len(sorted_objects), n_clusters)
        batches = np.array_split(sorted_objects, splits)

        return batches, n_complete_splits


def run_aba(X, n_anticlusters, categories=None, small_anticlusters_flag=False):

    # Convert X to float numpy array
    X = np.array(X).astype(float)

    # Check if n_clusters is a list, tuple or ndarray
    if isinstance(n_anticlusters, list) or isinstance(n_anticlusters, tuple) or isinstance(n_anticlusters, np.ndarray):
        if len(n_anticlusters) > 1:
            n_anticlusters_list = n_anticlusters[1:]
            n_anticlusters = n_anticlusters[0]
        else:
            n_anticlusters_list = []
            n_anticlusters = n_anticlusters[0]
    else:
        n_anticlusters_list = []

    # Get number of objects
    n_objects = X.shape[0]

    # Initialize labels
    labels = np.full(n_objects, -1)

    # Compute distances to global center
    global_center = X.mean(axis=0)
    distances = cdist(X, [global_center], 'sqeuclidean')

    # Sort objects in descending distance from global center
    sorted_objects = np.argsort(-distances[:, 0])

    # Get batches
    if categories is None:
        batches = get_batches(sorted_objects, n_anticlusters, small_anticlusters_flag=small_anticlusters_flag)
    else:
        batches, n_complete_batches = get_batches(sorted_objects, n_anticlusters, categories[sorted_objects])

        # Get unique categories and the counts for each category 
        unique_categories, category_counts = np.unique(categories, return_counts=True)
        
        # Get number of unique categories
        n_categories = len(unique_categories)

        # Initialize cluster category counts
        cluster_category_counts = np.zeros((n_anticlusters, n_categories))

        # Compute upper bounds on category counts
        category_upper_bounds = np.ceil(category_counts / n_anticlusters)

        # Update cluster_category_counts with first batch
        cluster_category_counts[np.arange(n_anticlusters), categories[batches[0]]] += 1

    # Initialize centers
    centers = X[batches[0], :]

    # Assign first objects to centers
    labels[batches[0]] = np.arange(n_anticlusters)

    # Process batches
    for i, batch in enumerate(batches[1:]):

        # Assign objects
        if categories is None:
            batch_labels = assign_objects(X, centers, batch)
        else: 
            if i >= n_complete_batches:
                batch_labels = assign_objects(X, centers, batch, categories[batch], category_upper_bounds, cluster_category_counts)
            else:
                batch_labels = assign_objects(X, centers, batch)

            # Update categories counter
            cluster_category_counts[batch_labels, categories[batch]] += 1

        # Update centers
        diff = X[batch, :] - centers[batch_labels, :]
        centers[batch_labels, :] += diff * (1 / (i + 2))

        # Update labels
        labels[batch] = batch_labels

    # Perform recursive partitioning (if needed)
    if len(n_anticlusters_list) > 0:

        # Initialize counter
        counter = 0

        # Initialize new labels
        new_labels = np.full(n_objects, -1)

        # Run in parallel
        results = Parallel(n_jobs=-1, backend='threading')(
            delayed(process_cluster)(cluster, labels, X, n_anticlusters_list, categories) for cluster in range(n_anticlusters)
        )

        for cluster, cluster_members, labels_ in results:
            new_labels[cluster_members] = counter + labels_
            counter += np.prod(n_anticlusters_list)

        labels = new_labels

    return labels



