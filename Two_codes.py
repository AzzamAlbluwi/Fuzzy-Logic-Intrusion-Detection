import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.spatial import distance
from scipy.spatial.distance import mahalanobis
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
import seaborn as sns

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

# Step 5: Calculate the Mahalanobis distance from each testing sample to each cluster
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

# Step 6: Extract input and output information and save them to separate CSV files
value = 1.5
input_variables = np.hstack((distances[:, 0].reshape(-1, 1), distances[:, 1].reshape(-1, 1)))
pd.DataFrame(input_variables).to_csv('input_variables_4_pca_36k_py.csv', index=False, header=False)

# Step 7: Load the input data from the CSV file
input_data = pd.read_csv('input_variables_4_pca_36K_py.csv', header=None)
num_rows, _ = input_data.shape

# Step 8: Define the Fuzzy Logic variables and membership functions
# Input Variables
input_var_1 = ctrl.Antecedent(np.arange(0, 15.001, 0.001), 'input_var_1')
input_var_1['low'] = fuzz.trimf(input_var_1.universe, [0, 0, 7.5])
input_var_1['medium'] = fuzz.trimf(input_var_1.universe, [0, 7.5, 15])
input_var_1['high'] = fuzz.trimf(input_var_1.universe, [7.5, 15, 15])

input_var_2 = ctrl.Antecedent(np.arange(0, 2.001, 0.001), 'input_var_2')
input_var_2['low'] = fuzz.trimf(input_var_2.universe, [0, 0, 1])
input_var_2['medium'] = fuzz.trimf(input_var_2.universe, [0, 1, 2])
input_var_2['high'] = fuzz.trimf(input_var_2.universe, [1, 2, 2])

# Output Variable
output_var = ctrl.Consequent(np.arange(1, 3.001, 1), 'output_var')
output_var['normal'] = fuzz.trimf(output_var.universe, [1, 1, 2])
output_var['attack'] = fuzz.trimf(output_var.universe, [1, 2, 3])



# Step 9: Define the Fuzzy Rules
rule1 = ctrl.Rule(input_var_1['low'] & input_var_2['low'], output_var['normal'])
rule2 = ctrl.Rule(input_var_1['low'] & input_var_2['medium'], output_var['normal'])
rule3 = ctrl.Rule(input_var_1['low'] & input_var_2['high'], output_var['attack'])
rule4 = ctrl.Rule(input_var_1['medium'] & input_var_2['low'], output_var['normal'])
rule5 = ctrl.Rule(input_var_1['medium'] & input_var_2['medium'], output_var['normal'])
rule6 = ctrl.Rule(input_var_1['medium'] & input_var_2['high'], output_var['attack'])
rule7 = ctrl.Rule(input_var_1['high'] & input_var_2['low'], output_var['normal'])
rule8 = ctrl.Rule(input_var_1['high'] & input_var_2['medium'], output_var['attack'])
rule9 = ctrl.Rule(input_var_1['high'] & input_var_2['high'], output_var['attack'])



# Step 10: Create the Fuzzy Control System
fuzzy_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
fuzzy_sim = ctrl.ControlSystemSimulation(fuzzy_ctrl)

# Step 11: Loop through each row of input data, make predictions, apply threshold, and save results
results = []
for i in range(num_rows):
    # Get the input values for the current data point
    input_values = input_data.iloc[i].values
    input_var_1_val, input_var_2_val = input_values

    # Fuzzify the input variables
    fuzzy_sim.input['input_var_1'] = input_var_1_val
    fuzzy_sim.input['input_var_2'] = input_var_2_val

    # Apply the fuzzy rules and defuzzify to get the predicted class (continuous value)
    fuzzy_sim.compute()
    predicted_class = fuzzy_sim.output['output_var']

    # Apply threshold to convert the continuous value to discrete class label
    discrete_class = 'Attack' if predicted_class <= value else 'Normal'
    results.append(discrete_class)

# Step 12: Save the results to a CSV file without the header
output_data = pd.DataFrame(results, columns=['Predicted Class'])
output_data.to_csv('output_results.csv', index=False, header=False)

# Step 13: Load the true class labels from the 'true_values.csv' file
true_labels = ['Normal'] * 18000 + ['Attack'] * 17999

print(f"value: {value}")

# Step 14: Calculate the accuracy using scikit-learn
output_labels = output_data.iloc[:, 0].values
accuracy = accuracy_score(true_labels, output_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Step 15: Create the confusion matrix
cm = confusion_matrix(true_labels, output_labels, labels=['Normal', 'Attack'])

# Step 16: Plot the confusion matrix as a heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title(f'Confusion Matrix\nAccuracy: {accuracy * 100:.2f}%')
plt.show()


# Step 14: Calculate the accuracy using scikit-learn
output_labels = output_data.iloc[:, 0].values
accuracy = accuracy_score(true_labels, output_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Create the confusion matrix
cm = confusion_matrix(true_labels, output_labels, labels=['Normal', 'Attack'])

# Extract values from the confusion matrix
true_negatives, false_positives, false_negatives, true_positives = cm.ravel()

# Calculate true positive percentage
tpp = true_positives / (true_positives + false_negatives) * 100

# Calculate false positive percentage
fpp = false_positives / (false_positives + true_negatives) * 100

# Calculate true negative percentage
tnp = true_negatives / (true_negatives + false_positives) * 100

# Calculate false negative percentage
fnp = false_negatives / (false_negatives + true_positives) * 100


# Plot the confusion matrix as a heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title(f'Confusion Matrix\nTrue Positive Percentage: {tpp:.2f}%\nFalse Positive Percentage: {fpp:.2f}%')
plt.show()

# Print the results
print(f"True Positive Percentage (TPP): {tpp:.2f}%")
print(f"False Positive Percentage (FPP): {fpp:.2f}%")
print(f"True Negative Percentage (TNP): {tnp:.2f}%")
print(f"False Negative Percentage (FNP): {fnp:.2f}%")