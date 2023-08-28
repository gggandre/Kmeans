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




# Run code

1. Clone the repo or download the .zip
![image](https://github.com/gggandre/Kmeans/assets/84719490/76238fa3-cb0a-4e34-b287-55480d9a5911)

2. Open the folder in your local environment (make sure you have Python installed, along with the libraries numpy, pandas, and tkinter)
![image](https://github.com/gggandre/Kmeans/assets/84719490/e6e12091-ba80-4bc5-a31a-880cbd86f95f)

- Note: If the libraries are not installed, use ```pip install pandas numpy tkinter```
![image](https://github.com/gggandre/Kmeans/assets/84719490/a1b79a95-7bb6-41f6-ab24-3e186a5ea8bd)

4. Run the code with code runner or terminal
![image](https://github.com/gggandre/Kmeans/assets/84719490/92eecb31-649d-417b-8fdc-59dc7d533675)
![image](https://github.com/gggandre/Kmeans/assets/84719490/829a7fbd-93c0-4fd4-bc85-84a653138cca)

6. Use the GUI to write the number of clusters (k)
![image](https://github.com/gggandre/Kmeans/assets/84719490/d576ce3a-fb38-449c-a8b1-9ebe3787a7f5)

7. Once you write the number of clusters, click in the button "Run KMeans"
![image](https://github.com/gggandre/Kmeans/assets/84719490/053e0dd1-a3e2-43c2-af2f-094617691d55)

8. Now you see the results
![image](https://github.com/gggandre/Kmeans/assets/84719490/828d4931-b7de-4ba6-9725-4915ba5eb46b)

 - Note: If you exceed the number of clusters (k) the program show a error message
![image](https://github.com/gggandre/Kmeans/assets/84719490/8fd79d00-7979-40f9-aa36-b5e9ddcdcaef)

