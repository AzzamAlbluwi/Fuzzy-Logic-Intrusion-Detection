import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Step 1: Load the input data from the CSV file
# input_data = pd.read_csv('input_variables_30K.csv', header=None)
input_data = pd.read_csv('Datasets/Results/input_variables_4_pca.csv', header= None)

num_rows, num_cols = input_data.shape

data_value = 15000

# Step 2: Define the Fuzzy Logic variables and membership functions
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

# Step 3: Define the Fuzzy Rules
rule1 = ctrl.Rule(input_var_1['low'] & input_var_2['low'], output_var['normal'])
rule2 = ctrl.Rule(input_var_1['low'] & input_var_2['medium'], output_var['normal'])
rule3 = ctrl.Rule(input_var_1['low'] & input_var_2['high'], output_var['attack'])
rule4 = ctrl.Rule(input_var_1['medium'] & input_var_2['low'], output_var['normal'])
rule5 = ctrl.Rule(input_var_1['medium'] & input_var_2['medium'], output_var['normal'])
rule6 = ctrl.Rule(input_var_1['medium'] & input_var_2['high'], output_var['attack'])
rule7 = ctrl.Rule(input_var_1['high'] & input_var_2['low'], output_var['normal'])
rule8 = ctrl.Rule(input_var_1['high'] & input_var_2['medium'], output_var['attack'])
rule9 = ctrl.Rule(input_var_1['high'] & input_var_2['high'], output_var['attack'])

# Step 4: Create the Fuzzy Control System
fuzzy_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
fuzzy_sim = ctrl.ControlSystemSimulation(fuzzy_ctrl)

# Step 5: Loop through each row of input data, make predictions, apply threshold, and save results
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
    discrete_class = 'Attack' if predicted_class <= 1.5 else 'Normal'
    results.append(discrete_class)

    # Print the predicted class for the current data point
    # print(f"Data point {i+1}: Predicted class: {discrete_class}")

## Step 6: Save the results to a CSV file without the header
output_data = pd.DataFrame(results, columns=['Predicted Class'])
output_data.to_csv('output_results.csv', index=False, header=False)

# Step 7: Load the true class labels from the 'true_values.csv' file
# true_data = pd.read_csv('true_values.csv', header=None)
# true_labels = true_data.iloc[:, 0].values
true_labels = ['Normal'] * 15000 + ['Attack'] * 15000

# Step 8: Remove the header row from the 'output_results.csv' file
# output_data = pd.read_csv('output_results.csv', header=None)
output_labels = output_data.iloc[:, 0].values

# Step 9: Calculate the accuracy
correct_predictions = np.sum(output_labels == true_labels)
accuracy = correct_predictions / len(true_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Step 9: Calculate the accuracy using scikit-learn
accuracy = accuracy_score(true_labels, output_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Plot Input Variable 1
input_var_1.view()

# Plot Input Variable 2
input_var_2.view()

# Plot Output Variable
output_var.view()
plt.show()

# Step 7: Load the true class labels from the 'true_values.csv' file
# true_data = pd.read_csv('true_values.csv', header=None)
# true_labels = true_data.iloc[:, 0].values

# # Step 8: Calculate the accuracy
# correct_predictions = np.sum(output_data['Predicted Class'] == true_labels)
# accuracy = correct_predictions / num_rows
# print(f"Accuracy: {accuracy * 100:.2f}%")




# Step 7: Create a file containing the true values (first 1000 normal, next 1000 attack)
# true_values = ['Normal'] * 1000 + ['Attack'] * 15000
# true_data = pd.DataFrame(true_values, columns=['True Class'])
# true_data.to_csv('true_values.csv', index=False)

# # Step 8: Calculate Accuracy
# true_data = pd.read_csv('true_values.csv')
# output_data = pd.read_csv('output_results.csv')

# true_classes = true_data['True Class'].values
# predicted_classes = output_data['Predicted Class'].values

# accuracy = np.mean(true_classes == ['Normal' if p == 1 else 'Attack' for p in predicted_classes])
# print(f"Accuracy: {accuracy * 100:.2f}%")
