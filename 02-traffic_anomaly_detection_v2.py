import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.spatial import distance
from scipy.spatial.distance import mahalanobis

# Step 1: Load the necessary datasets and remove headers
data = pd.read_csv('Datasets/Results/traffic_normal_attacks_36K.csv', header=1).to_numpy()
testing_data = pd.read_csv('Datasets/Results/traffic_normal_attacks_test_36K.csv', header=1).to_numpy()

# Step 2: Perform PCA on the dataset
num_components = 5
pca = PCA(n_components=num_components)
reduced_data = pca.fit_transform(data)
reduced_testing_data = np.dot(testing_data - np.mean(data, axis=0), pca.components_[:num_components].T)

# Step 3: Divide the reduced dataset into separate clusters
num_normal_samples = 18000
num_attack_samples = 18000

normal_data = reduced_data[:num_normal_samples, :]
attack_data = reduced_data[num_normal_samples:num_normal_samples + num_attack_samples, :]

clusters = [normal_data, attack_data]  # Create clusters as a list

# Step 4: Fit Gaussian Mixture Models (GMMs) to each cluster
num_clusters = len(clusters)
num_gmms = 1
gmm_models = []

for i in range(num_clusters):
    for j in range(num_gmms):
        gmm = GaussianMixture(n_components=1, covariance_type='full')
        gmm.fit(clusters[i])
        gmm_models.append(gmm)

# # Step 5: Calculate the Mahalanobis distance from each testing sample to each cluster
num_testing_samples = reduced_testing_data.shape[0]
distances = np.zeros((num_testing_samples, num_clusters))

for i in range(num_testing_samples):
    for j in range(num_clusters):
        cluster_distances = np.zeros(num_gmms)
        for k in range(num_gmms):
            log_likelihood = gmm_models[j * num_gmms + k].score_samples(reduced_testing_data[i, :].reshape(1, -1))
            cov_inv = np.linalg.inv(gmm_models[j * num_gmms + k].covariances_[0])
            diff = reduced_testing_data[i, :] - gmm_models[j * num_gmms + k].means_[0]
            cluster_distances[k] = np.sqrt(np.dot(np.dot(diff, cov_inv), diff.T))
        distances[i, j] = np.min(cluster_distances)

# Step 5: Calculate the Mahalanobis distance from each testing sample to each cluster
# num_testing_samples = reduced_testing_data.shape[0]
# distances = np.zeros((num_testing_samples, num_clusters))

# for i in range(num_testing_samples):
#     for j in range(num_clusters):
#         cluster_distances = np.zeros(num_gmms)
#         for k in range(num_gmms):
#             cluster_distances[k] = mahalanobis(reduced_testing_data[i, :], gmm_models[j * num_gmms + k].means_[0], np.linalg.inv(gmm_models[j * num_gmms + k].covariances_[0]))
#         distances[i, j] = np.min(cluster_distances)

# Step 6: Extract input and output information and save them to separate CSV files
# You can replace "input_variables.csv" with an appropriate filename
input_variables = np.hstack((distances[:, 0].reshape(-1, 1), distances[:, 1].reshape(-1, 1)))
pd.DataFrame(input_variables).to_csv('input_variables_4_pca_36k_py.csv', index=False, header=False)
