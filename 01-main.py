# Import necessary functions from the 'functions' module
from functions import read_csv_file, drop_col, save_data_to_csv, check_null_values, drop_columns, normalize_data, select_random_rows
import pandas as pd

# Read the dataset from the CSV file named "UNSW_NB15.csv"
dataset = read_csv_file("all")

# Drop the "id" column from the dataset
dataset = drop_columns(dataset, "id")

# Filter the dataset based on the attack categories
normal_traffic = dataset[dataset['attack_cat'] == 'Normal']
generic_traffic = dataset[dataset['attack_cat'] == 'Generic']
exploits_traffic = dataset[dataset['attack_cat'] == 'Exploits']
fuzzers_traffic = dataset[dataset['attack_cat'] == 'Fuzzers']

# Check for null values in the 'normal_traffic' dataset
check_null_values(normal_traffic)
# Check for null values in the 'generic_traffic' dataset
check_null_values(generic_traffic)
# Check for null values in the 'exploits_traffic' dataset
check_null_values(exploits_traffic)
# Check for null values in the 'fuzzers_traffic' dataset
check_null_values(fuzzers_traffic)


# Select the non-numerical columns and time related columns 
non_numerical_cols = ['proto', 'service', 'state', 'sbytes', 'sinpkt', 'Dintpkt', 
'dinpkt', 'sjit', 'djit', 'tcprtt', 'synack', 'ackdat', 'attack_cat', 'label']

# Drop the non-numerical columns for each dataset
normal_traffic = drop_columns(normal_traffic, non_numerical_cols)
generic_traffic = drop_columns(generic_traffic, non_numerical_cols)
exploits_traffic = drop_columns(exploits_traffic, non_numerical_cols)
fuzzers_traffic = drop_columns(fuzzers_traffic, non_numerical_cols)

# Normalize the data for each filtered dataset
# normal_traffic = normalize_data(normal_traffic)
# generic_traffic = normalize_data(generic_traffic)
# exploits_traffic = normalize_data(exploits_traffic)
# fuzzers_traffic = normalize_data(fuzzers_traffic)

print("Number of rows in normal_traffic:", normal_traffic.shape[0])
print("Number of rows in generic_traffic:", generic_traffic.shape[0])
print("Number of rows in exploits_traffic:", exploits_traffic.shape[0])
print("Number of rows in fuzzers_traffic:", fuzzers_traffic.shape[0])

# Select random rows for each filtered dataset
normal_traffic = select_random_rows(normal_traffic, 37000)
generic_traffic = select_random_rows(generic_traffic, 18000)
exploits_traffic = select_random_rows(exploits_traffic, 10000)
fuzzers_traffic = select_random_rows(fuzzers_traffic, 6000)

# print("Number of rows in normal_traffic:", normal_traffic.shape[0])
# print("Number of rows in generic_traffic:", generic_traffic.shape[0])
# print("Number of rows in exploits_traffic:", exploits_traffic.shape[0])
# print("Number of rows in fuzzers_traffic:", fuzzers_traffic.shape[0])

# testing_traffic = select_random_rows()
normal_traffic_test = select_random_rows(normal_traffic, 37000)
generic_traffic_test = select_random_rows(generic_traffic, 18000)
exploits_traffic_test = select_random_rows(exploits_traffic, 10000)
fuzzers_traffic_test = select_random_rows(fuzzers_traffic, 6000)

# traffic_test = pd.concat([normal_traffic_test, generic_traffic_test, exploits_traffic_test, fuzzers_traffic_test], ignore_index=True)
traffic_test = pd.concat([normal_traffic_test, generic_traffic_test], ignore_index=True)
save_data_to_csv(traffic_test, 'traffic_normal_attacks_test_55K.csv')


print("Number of rows in testing:", traffic_test.shape[0])

# Concatenate the dataframes vertically into one dataframe
# traffic = pd.concat([normal_traffic, generic_traffic, exploits_traffic, fuzzers_traffic], ignore_index=True)
traffic = pd.concat([normal_traffic, generic_traffic, exploits_traffic_test], ignore_index=True)

# Save the processed datasets to separate CSV files
save_data_to_csv(traffic, 'traffic_normal_attacks_55K.csv')

