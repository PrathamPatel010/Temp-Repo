import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, n_clusters, max_iterations=100):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations

    def initialize_centroids(self, X):
        centroids_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        centroids = X[centroids_indices]
        return centroids

    def closest_centroid(self, x, centroids):
        distances = np.linalg.norm(centroids - x, axis=1)
        closest_index = np.argmin(distances)
        return closest_index

    def update_centroids(self, clusters, X):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for cluster_index in range(self.n_clusters):
            cluster_points = X[clusters == cluster_index]
            centroids[cluster_index] = np.mean(cluster_points, axis=0)
        return centroids

    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        for _ in range(self.max_iterations):
            clusters = np.array([self.closest_centroid(x, self.centroids) for x in X])
            new_centroids = self.update_centroids(clusters, X)
            if np.allclose(new_centroids, self.centroids):
                break
            self.centroids = new_centroids
        return clusters


# Generate sample data
np.random.seed(0)
X = np.random.rand(100, 2)

# Number of clusters
k = 3

# Initialize and fit KMeans model
kmeans = KMeans(n_clusters=k)
clusters = kmeans.fit(X)

# Plotting the clusters
for i in range(k):
    cluster_points = X[clusters == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}')

plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], marker='x', color='black', label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()