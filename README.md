# Kmeans

## Initialization:
- A number k is set, representing the desired number of clusters.
- k random points from the dataset are chosen as the initial centroids.

## Assignment of Points to Centroids:
- For each point in the dataset, the distance to each of the k centroids is calculated.
- Each point is assigned to the nearest centroid.

## Recalculation of Centroids:
- Once all points have been assigned to a centroid, the centroids are recalculated.
- The new centroid is simply the average of all points assigned to that cluster.
- If any centroid has no assigned points (which is rare but can happen), it is reset by choosing a random point from the dataset as the new centroid.

## Convergence:
- These steps are repeated until centroids no longer change between consecutive iterations or until a maximum number of iterations is reached.
- When the algorithm converges, we have our final clusters.
