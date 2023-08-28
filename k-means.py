#----------------------------------------------------------
# Feedback Moment: Module 2
# Implementation of a machine learning technique without the use of a framework.
#
# Date: 27-Aug-2023
# Author:
#           A01753176 Gilberto André García Gaytán
#----------------------------------------------------------
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
import tkinter.messagebox

def euclidean_distance(point1, point2):
    """
    The function calculates the Euclidean distance between two points in a multi-dimensional space.
    :param point1: The first point in the Euclidean distance calculation. It can be a list, tuple, or
    numpy array representing the coordinates of the point in n-dimensional space
    :param point2: The first point in the Euclidean distance calculation. It is a numpy array
    representing a point in n-dimensional space
    :return: the Euclidean distance between two points.
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Kmeans algorithm
class KMeans:
    def __init__(self, k=3, max_iters=100, random_seed=None):
        """
        The function initializes the parameters for a k-means clustering algorithm.
        :param k: The number of clusters to create, defaults to 3 (optional)
        :param max_iters: The `max_iters` parameter specifies the maximum number of iterations that the
        algorithm will perform before stopping. This is used to control the convergence of the algorithm,
        defaults to 100 (optional)
        :param random_seed: The `random_seed` parameter is used to initialize the random number generator.
        By setting a specific value for `random_seed`, you can ensure that the random numbers generated
        during the execution of the code are reproducible. This can be useful for debugging or when you want
        to compare the results of different runs
        """
        self.k = k
        self.max_iters = max_iters
        self.random_seed = random_seed
        self.centroids = None

    def fit(self, data):
        """
        The `fit` function initializes centroids randomly and iteratively updates them until convergence
        using the k-means algorithm.
        :param data: The "data" parameter is a numpy array that represents the dataset on which the K-means
        algorithm will be applied. Each row of the array represents a data point, and each column represents
        a feature of that data point
        :return: The `fit` method returns the instance of the class itself (`self`).
        """
        np.random.seed(self.random_seed)
        random_indices = np.random.choice(len(data), self.k, replace=False)
        self.centroids = data[random_indices, :]
        for _ in range(self.max_iters):
            labels = np.array([self._closest_centroid(point) for point in data])
            new_centroids = []
            for i in range(self.k):
                if (labels == i).any():
                    new_centroids.append(data[labels == i].mean(axis=0))
                else:
                    # Reinicializar el centroide a una nueva posición aleatoria si no tiene puntos asignados
                    new_centroids.append(data[np.random.choice(len(data))])
            new_centroids = np.array(new_centroids)
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids
        return self

    def predict(self, data):
        """
        The `predict` function takes in a dataset and returns an array of the closest centroid for each data
        point.
        :param data: The `data` parameter is a list or array of data points that you want to make
        predictions for. Each data point should be a list or array of features
        :return: a numpy array containing the closest centroid for each point in the input data.
        """
        return np.array([self._closest_centroid(point) for point in data])

    def _closest_centroid(self, point):
        """
        The function `_closest_centroid` calculates the Euclidean distance between a given point and each
        centroid in a list, and returns the index of the centroid with the minimum distance.
        :param point: The point parameter represents a data point for which we want to find the closest
        centroid
        :return: the index of the closest centroid to the given point.
        """
        distances = [euclidean_distance(point, centroid) for centroid in self.centroids]
        return np.argmin(distances)

# GUI para KMeans
class KMeansGUI(tk.Tk):
    def __init__(self, data, features):
        """
        The above function is the initialization function for a KMeans Clustering GUI application using
        Tkinter in Python.
        :param data: The "data" parameter is the dataset that will be used for clustering. It should be a
        matrix or dataframe where each row represents a data point and each column represents a feature
        of that data point
        :param features: The "features" parameter is a list that contains the names of the features or
        attributes of the data. These features are used to perform the KMeans clustering algorithm on the
        Spotify data
        """
        super().__init__()
        self.data = data
        self.features = features
        self.kmeans_model = None
        self.title("KMeans Clustering - Spotify Data")
        tk.Label(self, text="Number of Clusters (k):").pack(pady=10)
        self.k_entry = tk.Entry(self)
        self.k_entry.pack(pady=10)
        self.run_button = tk.Button(self, text="Run KMeans", command=self.run_kmeans)
        self.run_button.pack(pady=20)
        self.tree = ttk.Treeview(self, columns=('Song', 'Cluster'))
        self.tree.heading('#0', text='Index')
        self.tree.heading('Song', text='Song')
        self.tree.heading('Cluster', text='Cluster')
        self.tree.pack(pady=20, padx=20, expand=True, fill='both')

def run_kmeans(self):
    """
    The function `run_kmeans` performs K-means clustering on a set of features and displays the results
    in a treeview widget. Before running, it checks if the entered k value exceeds the number of data points.
    """
    k = int(self.k_entry.get())
    
    # Check if k exceeds the number of data points
    if k >= len(self.features):
        # Display an error message to the user
        error_msg = "Entered k value exceeds the number of data points. Please try with a smaller value."
        tk.messagebox.showerror("Error", error_msg)
        return
    
    self.kmeans_model = KMeans(k=k, random_seed=42)
    self.kmeans_model.fit(self.features)
    clusters = self.kmeans_model.predict(self.features)
    for i in self.tree.get_children():
        self.tree.delete(i)
    for idx, (song_name, cluster) in enumerate(zip(self.data['track_name'], clusters)):
        self.tree.insert('', 'end', text=str(idx+1), values=(song_name, cluster))
KMeansGUI.run_kmeans = run_kmeans

# Load the data and select the features
# If you run your code in your local environment you must change the path to the dataset to your own path.
data = pd.read_csv('D:\ia_1\ml\evidencia\spotify-2023.csv', encoding="ISO-8859-1")
selected_features = ['bpm', 'danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 
                    'instrumentalness_%', 'liveness_%', 'speechiness_%']
X = data[selected_features].values
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Create and execute the GUI application
app = KMeansGUI(data, X)
app.mainloop()
