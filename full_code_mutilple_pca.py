# Import necessary libraries
from functions import read_csv_file, drop_col, save_data_to_csv, check_null_values, drop_columns, normalize_data, select_random_rows, split_train_test
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



number_data = 9000  # Replace this with the desired number of rows to select
num_testing = 9000

value = 1.5

# Define the range of PCA components to try
min_components = 1
max_components = 8

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
# normal_traffic = drop_columns(normal_traffic, non_numerical_cols)
# generic_traffic = drop_columns(generic_traffic, non_numerical_cols)

normal_traffic_data = drop_columns(normal_traffic, non_numerical_cols)
generic_traffic_data = drop_columns(generic_traffic, non_numerical_cols)


# Using .iloc to select the first 'num_rows_to_select' rows from `normal_traffic`
normal_traffic = normal_traffic_data.iloc[:number_data]
normal_traffic_test = normal_traffic_data.iloc[number_data:number_data + number_data]

generic_traffic = generic_traffic_data.iloc[:number_data]
generic_traffic_test = generic_traffic_data.iloc[number_data:number_data + number_data]

# Concatenate the filtered datasets
traffic = pd.concat([normal_traffic, generic_traffic], ignore_index=True)
traffic_test = pd.concat([normal_traffic_test, generic_traffic_test], ignore_index=True)



best_accuracy = 0.0
best_num_components = 0
best_threshold = 0.0
best_cm = None

# Lists to store evaluation metrics for each configuration
# Calculate the number of training and testing data points
num_data = []
pca_components_list = []
accuracy_list = []
tpp_list = []
fpp_list = []
tnp_list = []
fnp_list = []
precision_list = []
recall_list = []
f_score_list = []


# Loop through different numbers of PCA components
for num_components in range(min_components, max_components + 1):
    # Step 2: Perform PCA on the dataset
    pca = PCA(n_components=num_components)
    reduced_data = pca.fit_transform(traffic)
    reduced_testing_data = np.dot(traffic_test - np.mean(traffic, axis=0), pca.components_[:num_components].T)

    # Step 3: Divide the reduced dataset into separate clusters
    num_normal_samples = number_data
    num_attack_samples = number_data
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
    # value = 1.5
    # input_variables = np.hstack((distances[:, 0].reshape(-1, 1), distances[:, 1].reshape(-1, 1)))
    # pd.DataFrame(input_variables).to_csv(f'input_variables_{num_components}_pca_36k_py.csv', index=False, header=False)

    # # Step 7: Load the input data from the CSV file
    # input_data = pd.read_csv(f'input_variables_{num_components}_pca_36K_py.csv', header=None)
    # num_rows, _ = input_data.shape

    # Step 6: Extract input and output information

    input_variables = np.hstack((distances[:, 0].reshape(-1, 1), distances[:, 1].reshape(-1, 1)))

    # Convert the input variables to a DataFrame
    input_data = pd.DataFrame(input_variables, columns=['Input_Var_1', 'Input_Var_2'])

    # Step 7: Get the number of rows in the input data
    num_rows = input_data.shape[0]

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

    # Load the true class labels from the 'true_values.csv' file
    true_labels = ['Normal'] * num_testing + ['Attack'] * num_testing

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

    # Step 12: Calculate the accuracy using scikit-learn
    output_data = pd.DataFrame(results, columns=['Predicted Class'])
    output_labels = output_data.iloc[:, 0].values
    accuracy = accuracy_score(true_labels, output_labels)

    # Step 13: Create the confusion matrix
    cm = confusion_matrix(true_labels, output_labels, labels=['Normal', 'Attack'])

    # Extract values from the confusion matrix
    true_negatives, false_positives, false_negatives, true_positives = cm.ravel()

    # Calculate true positive percentage
    tpp = true_positives / (true_positives + false_negatives)

    # Calculate false positive percentage
    fpp = false_positives / (false_positives + true_negatives)

    # Calculate true negative percentage
    tnp = true_negatives / (true_negatives + false_positives)

    # Calculate false negative percentage
    fnp = false_negatives / (false_negatives + true_positives)
    
    # Calculate Precision
    precision = true_positives / (true_positives + false_positives)

    # Calculate Recall (True Positive Rate)
    recall = true_positives / (true_positives + false_negatives)

    # Calculate F-score
    f_score = 2 * (precision * recall) / (precision + recall)

    print(f"PCA Components: {num_components}")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    # Print the results
    print(f"True Positive Percentage (TPP): {tpp * 100:.2f}%")
    print(f"False Positive Percentage (FPP): {fpp * 100:.2f}%")
    print(f"True Negative Percentage (TNP): {tnp * 100:.2f}%")
    print(f"False Negative Percentage (FNP): {fnp * 100:.2f}%") 
    print(f"Precision: {precision:.4f}")
    print(f"Recall (True Positive Rate): {recall:.4f}")
    print(f"F-score: {f_score:.4f}")

    # Store the best configuration if it has the highest accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_num_components = num_components
        best_cm = cm

    # Calculate evaluation metrics
    num_data.append(number_data*2)
    pca_components_list.append(num_components)
    accuracy_list.append(accuracy)
    tpp_list.append(tpp)
    fpp_list.append(fpp)
    tnp_list.append(tnp)
    fnp_list.append(fnp)
    precision_list.append(precision)
    recall_list.append(recall)
    f_score_list.append(f_score)


# Print the best configuration and accuracy
print("-------------------- Best Results -------------------")
print(f"Best Configuration - PCA Components: {best_num_components}")
print(f"Best Accuracy: {best_accuracy * 100:.2f}%")

# Extract values from the best confusion matrix
best_true_negatives, best_false_positives, best_false_negatives, best_true_positives = best_cm.ravel()

# Calculate true positive percentage for the best configuration
best_tpp = best_true_positives / (best_true_positives + best_false_negatives)

# Calculate false positive percentage for the best configuration
best_fpp = best_false_positives / (best_false_positives + best_true_negatives)

# Calculate true negative percentage for the best configuration
best_tnp = best_true_negatives / (best_true_negatives + best_false_positives) 

# Calculate false negative percentage for the best configuration
best_fnp = best_false_negatives / (best_false_negatives + best_true_positives) 

# Calculate Precision
precision = true_positives / (true_positives + false_positives) 

# Calculate Recall (True Positive Rate)
recall = true_positives / (true_positives + false_negatives)

# Calculate F-score
f_score = 2 * ((precision * recall) / (precision + recall))

# Print the best results
print("-----------------------------------------------------")
print(f"Best True Positive Percentage (TPP): {best_tpp * 100:.2f}%")
print(f"Best False Positive Percentage (FPP): {best_fpp * 100:.2f}%")
print(f"Best True Negative Percentage (TNP): {best_tnp * 100:.2f}%")
print(f"Best False Negative Percentage (FNP): {best_fnp * 100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall (True Positive Rate): {recall:.4f}")
print(f"F-score: {f_score:.4f}")



# Create a DataFrame to store the evaluation metrics
evaluation_df = pd.DataFrame({
    'Training/Testing No.': num_data,
    'PCA': pca_components_list,
    'Accuracy (%)': accuracy_list,
    'TPP': tpp_list,
    'FPP': fpp_list,
    'TNP': tnp_list,
    'FNP': fnp_list,
    'Precision': precision_list,
    'Recall (TPR)': recall_list,
    'F-score': f_score_list
})

# Save the DataFrame to a CSV file
evaluation_df.to_csv('evaluation_metrics.csv', index=False)

# Plot the best confusion matrix as a heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title(f'Best Confusion Matrix\nBest Accuracy: {best_accuracy * 100:.2f}%')
plt.show()

# Plot the evaluation metrics
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.plot(pca_components_list, accuracy_list, marker='o')
plt.xlabel('Number of PCA Components')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs. Number of PCA Components')

plt.subplot(2, 2, 2)
plt.plot(pca_components_list, tpp_list, marker='o', label='True Positive Percentage (TPP)')
plt.plot(pca_components_list, fpp_list, marker='o', label='False Positive Percentage (FPP)')
plt.xlabel('Number of PCA Components')
plt.ylabel('Percentage')
plt.legend()
plt.title('TPP and FPP vs. Number of PCA Components')

plt.subplot(2, 2, 3)
plt.plot(pca_components_list, tnp_list, marker='o', label='True Negative Percentage (TNP)')
plt.plot(pca_components_list, fnp_list, marker='o', label='False Negative Percentage (FNP)')
plt.xlabel('Number of PCA Components')
plt.ylabel('Percentage')
plt.legend()
plt.title('TNP and FNP vs. Number of PCA Components')

plt.tight_layout()
plt.show()
