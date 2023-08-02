# Import necessary libraries
from functions import read_csv_file, drop_col, save_data_to_csv, check_null_values, drop_columns, normalize_data, select_random_rows
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import mahalanobis
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Read the dataset from the CSV file named "UNSW_NB15.csv"
dataset = read_csv_file("all")

# Drop the "id" column from the dataset
dataset = drop_columns(dataset, "id")

# Filter the dataset based on the attack categories
normal_traffic = dataset[dataset['attack_cat'] == 'Normal']
generic_traffic = dataset[dataset['attack_cat'] == 'Generic']

# Check for null values in the 'normal_traffic' dataset
check_null_values(normal_traffic)
# Check for null values in the 'generic_traffic' dataset
check_null_values(generic_traffic)

# Select the non-numerical columns and time-related columns 
non_numerical_cols = ['proto', 'service', 'state', 'sbytes', 'sinpkt', 'Dintpkt', 
                      'dinpkt', 'sjit', 'djit', 'tcprtt', 'synack', 'ackdat', 'attack_cat', 'label']

# Drop the non-numerical columns for each dataset
normal_traffic = drop_columns(normal_traffic, non_numerical_cols)
generic_traffic = drop_columns(generic_traffic, non_numerical_cols)

# Select random rows for each filtered dataset
normal_traffic = select_random_rows(normal_traffic, 18000)
generic_traffic = select_random_rows(generic_traffic, 18000)

# testing_traffic = select_random_rows()  # This line seems to be commented out, no action taken

# Select random rows for the test datasets
normal_traffic_test = select_random_rows(normal_traffic, 18000)
generic_traffic_test = select_random_rows(generic_traffic, 18000)

# Concatenate the filtered datasets
traffic = pd.concat([normal_traffic, generic_traffic], ignore_index=True)
traffic_test = pd.concat([normal_traffic_test, generic_traffic_test], ignore_index=True)

"""
Code 2
"""

# Step 2: Perform PCA on the dataset
num_components = 5
pca = PCA(n_components=num_components)
reduced_data = pca.fit_transform(traffic)
reduced_testing_data = np.dot(traffic_test - np.mean(traffic, axis=0), pca.components_[:num_components].T)

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
# Define the Fuzzy Antecedents (Input Variables) and their membership functions
input_var_1 = ctrl.Antecedent(np.arange(0, 15.001, 0.001), 'input_var_1')
input_var_1['low'] = fuzz.trimf(input_var_1.universe, [0, 0, 7.5])
input_var_1['medium'] = fuzz.trimf(input_var_1.universe, [0, 7.5, 15])
input_var_1['high'] = fuzz.trimf(input_var_1.universe, [7.5, 15, 15])

input_var_2 = ctrl.Antecedent(np.arange(0, 2.001, 0.001), 'input_var_2')
input_var_2['low'] = fuzz.trimf(input_var_2.universe, [0, 0, 1])
input_var_2['medium'] = fuzz.trimf(input_var_2.universe, [0, 1, 2])
input_var_2['high'] = fuzz.trimf(input_var_2.universe, [1, 2, 2])

# Output Variable
# Define the Fuzzy Consequent (Output Variable) and its membership functions
output_var = ctrl.Consequent(np.arange(1, 3.001, 1), 'output_var')
output_var['normal'] = fuzz.trimf(output_var.universe, [1, 1, 2])
output_var['attack'] = fuzz.trimf(output_var.universe, [1, 2, 3])

# Step 9: Define the Fuzzy Rules
# Define the Fuzzy Rules using the Antecedents and Consequent defined above
# (Rules are in the form of IF-THEN)
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
# Combine the Fuzzy Rules to create the Fuzzy Control System
fuzzy_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
fuzzy_sim = ctrl.ControlSystemSimulation(fuzzy_ctrl)

# Step 11: Loop through each row of input data, make predictions, apply threshold, and save results
# Make predictions using the Fuzzy Control System, apply a threshold, and store the results
results = []
for i in range(num_rows):
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

# Step 13: Load the true class labels from the 'true_values.csv' file
# Load the true class labels to evaluate the accuracy of the model
true_labels = ['Normal'] * 18000 + ['Attack'] * 18000

# Step 14: Calculate the accuracy using scikit-learn
# Calculate the accuracy of the model using scikit-learn's accuracy_score function
output_data = pd.DataFrame(results, columns=['Predicted Class'])
output_labels = output_data.iloc[:, 0].values
accuracy = accuracy_score(true_labels, output_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Step 15: Create the confusion matrix
# Create the confusion matrix to evaluate the model's performance
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

# Print the results
print(f"True Positive Percentage (TPP): {tpp:.2f}%")
print(f"False Positive Percentage (FPP): {fpp:.2f}%")
print(f"True Negative Percentage (TNP): {tnp:.2f}%")
print(f"False Negative Percentage (FNP): {fnp:.2f}%")

# Step 16: Plot the confusion matrix as a heatmap
# Visualize the confusion matrix using a heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title(f'Confusion Matrix\nAccuracy: {accuracy * 100:.2f}%')
plt.show()
