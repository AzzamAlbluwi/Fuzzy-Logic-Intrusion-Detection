import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# Step 1: Load the necessary datasets and remove headers
data = pd.read_csv('Datasets/traffic_normal_attacks.csv', header=1).values
testing_data = pd.read_csv('Datasets/Results/traffic_normal_attacks_test_30K.csv', header=1).values

# Step 2: Perform PCA on the dataset
num_components = 14
pca = PCA(n_components=num_components)
reduced_data = pca.fit_transform(data)
reduced_testing_data = (testing_data - np.mean(data, axis=0)) @ pca.components_[:num_components, :].T

# Step 3: Divide the reduced dataset into separate clusters
num_normal_samples = 15000
num_attack_samples = 15000

normal_data = reduced_data[:num_normal_samples, :]
attack_data = reduced_data[num_normal_samples:num_normal_samples + num_attack_samples, :]

# Combine normal and attack data into reduced_testing_data
reduced_testing_data = np.vstack((normal_data, attack_data))

# Update num_testing_samples
num_testing_samples = reduced_testing_data.shape[0]

# Step 4: Fit Gaussian Mixture Models (GMMs) to each cluster
num_clusters = len([normal_data, attack_data])
num_gmms = 1
gmm_models = [[GaussianMixture(n_components=1).fit(cluster)] for cluster in [normal_data, attack_data]]

# Step 5: Calculate the Mahalanobis distance from each testing sample to each cluster
distances = np.zeros((num_testing_samples, num_clusters))

for i in range(num_testing_samples):
    for j in range(num_clusters):
        cluster_distances = np.zeros(num_gmms)
        for k in range(num_gmms):
            # Calculate the log likelihood (negative Mahalanobis distance) using score_samples
            cluster_distances[k] = -gmm_models[j][k].score_samples([reduced_testing_data[i, :]])
        distances[i, j] = np.max(cluster_distances)  # Since Mahalanobis distance is negative log-likelihood

# Step 6: Determine the closest cluster for each testing sample
cluster_labels = np.argmin(distances, axis=1)

# Step 7: Calculate accuracy for each cluster
expected_labels = np.repeat(np.arange(num_clusters), num_testing_samples // num_clusters)
accuracies = np.zeros(num_clusters)

for i in range(num_clusters):
    # Calculate the start and end indices for each cluster
    start_idx = i * num_testing_samples // num_clusters
    end_idx = (i + 1) * num_testing_samples // num_clusters
    
    # Split the cluster_labels array into chunks for each cluster
    cluster_labels_chunk = cluster_labels[start_idx:end_idx]
    
    # Calculate cluster accuracy
    cluster_accuracy = np.mean(cluster_labels_chunk == i)
    accuracies[i] = cluster_accuracy

# Step 8: Calculate total accuracy
total_accuracy = np.mean(accuracies)

# Step 9: Print accuracies for each cluster
print('Accuracy for each cluster:')
for i in range(num_clusters):
    print(f'Cluster {i+1}: {accuracies[i] * 100:.2f}%')

# Step 10: Print total accuracy
print(f'Total Accuracy: {total_accuracy * 100:.2f}%')

# ... (previous code) ...

# Step 11: Save the distances, cluster_labels, and expected_labels to a CSV file named "distances.csv"
# Remove one row from distances to match the dimensions with expected_labels
# distances = distances[:-1]
# results = np.hstack((distances, cluster_labels.reshape(-1, 1), expected_labels.reshape(-1, 1)))
# np.savetxt('Datasets/Results/distances.csv', results, delimiter=',')


# Step 12: Extract input and output information and save them to separate CSV files
input_variables = distances[:, :2]  # Using the first two distances as input variables
np.savetxt('Datasets/Results/input_variables_py.csv', input_variables, delimiter=',')

output_variable = cluster_labels  # Use cluster_labels as output_variable
np.savetxt('Datasets/Results/output_variable_py.csv', output_variable, delimiter=',')
