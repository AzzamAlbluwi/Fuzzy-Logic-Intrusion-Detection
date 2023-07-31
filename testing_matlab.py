import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import numpy as np
from scipy.spatial import distance

# Step 1: Load the necessary datasets and remove headers
data = pd.read_csv('Datasets/traffic_normal_attacks.csv', header=1).values
testing_data = pd.read_csv('Datasets/Results/traffic_normal_attacks_test_30K.csv', header=1).values

# Step 2: Perform PCA on the dataset
num_components = 14  # Number of PCA components

pca = PCA(n_components=num_components)
pca.fit(data)
reduced_data = pca.transform(data)
reduced_testing_data = pca.transform(testing_data - data.mean(axis=0))

# Step 3: Divide the reduced dataset into separate clusters
num_normal_samples = 15000  # Number of normal traffic samples
num_attack_samples = 15000  # Number of attack traffic samples

normal_data = reduced_data[:num_normal_samples, :]
attack_data = reduced_data[num_normal_samples:num_normal_samples + num_attack_samples, :]

clusters = [normal_data, attack_data]  # Create clusters as a list

num_clusters = len(clusters)
num_gmms = 1  # Number of GMMs per cluster
gmm_models = [[None for j in range(num_gmms)] for i in range(num_clusters)]

for i in range(num_clusters):
    for j in range(num_gmms):
        gmm_models[i][j] = GaussianMixture(n_components=1, covariance_type='full')
        gmm_models[i][j].fit(clusters[i])


num_testing_samples = reduced_testing_data.shape[0]
distances = np.zeros((num_testing_samples, num_clusters))

for i in range(num_testing_samples):
    for j in range(num_clusters):
        cluster_distances = np.zeros(num_gmms)
        for k in range(num_gmms):
            cluster_mean = gmm_models[j][k].means_[0]  # Extract the mean as a 1-D vector
            cluster_covariance = gmm_models[j][k].covariances_[0]  # Extract the covariance as a 1-D vector
            cluster_distances[k] = distance.mahalanobis(reduced_testing_data[i], cluster_mean, np.linalg.inv(cluster_covariance))
        distances[i, j] = np.min(cluster_distances)

cluster_labels = np.argmin(distances, axis=1)

expected_labels = np.repeat(np.arange(1, num_clusters+1), num_testing_samples // num_clusters)
accuracies = np.zeros(num_clusters)

for i in range(1, num_clusters+1):
    cluster_indices = ((i-1) * num_testing_samples // num_clusters) + np.arange(num_testing_samples // num_clusters)
    cluster_accuracy = np.sum(cluster_labels[cluster_indices] == i) / (num_testing_samples // num_clusters)
    accuracies[i-1] = cluster_accuracy

total_accuracy = np.sum(accuracies) / num_clusters

print("Accuracy for each cluster:")
for i in range(1, num_clusters+1):
    print(f"Cluster {i}: {accuracies[i-1] * 100:.2f}%")




