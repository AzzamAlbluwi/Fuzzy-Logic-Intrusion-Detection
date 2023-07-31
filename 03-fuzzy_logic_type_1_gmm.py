import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load the input data from the CSV file
input_data = pd.read_csv('Datasets/Results/input_variables_4_pca.csv', header=None)
num_rows, num_cols = input_data.shape

data_value = 15000

# Step 2: Define the Fuzzy Logic variables and membership functions
# Input Variables
input_var_1 = ctrl.Antecedent(np.arange(0, 15.001, 0.001), 'input_var_1')
input_var_1['low'] = fuzz.gaussmf(input_var_1.universe, [0, 0, 7.5])
input_var_1['medium'] = fuzz.gaussmf(input_var_1.universe, [0, 7.5, 15])
input_var_1['high'] = fuzz.gaussmf(input_var_1.universe, [7.5, 15, 15])

input_var_2 = ctrl.Antecedent(np.arange(0, 2.001, 0.001), 'input_var_2')
input_var_2['low'] = fuzz.gaussmf(input_var_2.universe, [0, 0, 1])
input_var_2['medium'] = fuzz.gaussmf(input_var_2.universe, [0, 1, 2])
input_var_2['high'] = fuzz.gaussmf(input_var_2.universe, [1, 2, 2])

# Output Variable
output_var = ctrl.Consequent(np.arange(1, 3.001, 1), 'output_var')
output_var['normal'] = fuzz.gaussmf(output_var.universe, [1, 1, 2])
output_var['attack'] = fuzz.gaussmf(output_var.universe, [1, 2, 3])

# Step 3: Define the Fuzzy Rules (Type-2 fuzzy rules)
# Use fuzzy_and to handle repeated labels for the same antecedent
rule1 = ctrl.Rule(fuzz.fuzzy_and(input_var_1['low'], input_var_2['low']), output_var['normal'])
rule2 = ctrl.Rule(fuzz.fuzzy_and(input_var_1['low'], input_var_2['medium']), output_var['normal'])
rule3 = ctrl.Rule(fuzz.fuzzy_and(input_var_1['low'], input_var_2['high']), output_var['attack'])
rule4 = ctrl.Rule(fuzz.fuzzy_and(input_var_1['medium'], input_var_2['low']), output_var['normal'])
rule5 = ctrl.Rule(fuzz.fuzzy_and(input_var_1['medium'], input_var_2['medium']), output_var['normal'])
rule6 = ctrl.Rule(fuzz.fuzzy_and(input_var_1['medium'], input_var_2['high']), output_var['attack'])
rule7 = ctrl.Rule(fuzz.fuzzy_and(input_var_1['high'], input_var_2['low']), output_var['normal'])
rule8 = ctrl.Rule(fuzz.fuzzy_and(input_var_1['high'], input_var_2['medium']), output_var['attack'])
rule9 = ctrl.Rule(fuzz.fuzzy_and(input_var_1['high'], input_var_2['high']), output_var['attack'])

# Step 4: Create the Fuzzy Control System
fuzzy_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
fuzzy_sim = ctrl.ControlSystemSimulation(fuzzy_ctrl)


# The rest of the code remains the same as in the previous version.


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

# Step 6: Save the results to a CSV file without the header
output_data = pd.DataFrame(results, columns=['Predicted Class'])
output_data.to_csv('output_results.csv', index=False, header=False)

# Step 7: Load the true class labels from the provided list
true_labels = ['Normal'] * 15000 + ['Attack'] * 15000

# Step 8: Load the predicted class labels from the 'output_results.csv' file
output_data = pd.read_csv('output_results.csv', header=None)
output_labels = output_data.iloc[:, 0].values

# Step 9: Calculate the accuracy using scikit-learn
accuracy = accuracy_score(true_labels, output_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")
