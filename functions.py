# Import necessary libraries
import seaborn as sns  # visualization library
import matplotlib.pyplot as plt  # visualization library
import pandas as pd  # data manipulation library
import numpy as np  # numerical computation library
import os  # operating system related functions
from sklearn.decomposition import PCA  # principal component analysis library
from sklearn.preprocessing import MinMaxScaler  # scaling library
import random  # random number generator library
from io import StringIO  # input/output library for strings

def read_csv_file(file_type):
    # Define a dictionary containing the filenames for different file types
    filenames = {
        "all": "Datasets/UNSW_NB15.csv",
        "normal": "Datasets/normal_traffic.csv",
        "attacks": "Datasets/attacks_traffic.csv",
    }

    # Get the filename for the specified file type
    filename = filenames.get(file_type)

    # If the file type is not recognized, print an error message and return None
    if not filename:
        print("Invalid file type")
        return None

    # Read the data from the CSV file using pandas
    data = pd.read_csv(filename)

    # Return the data
    return data



def save_normalized_data_to_csv(data, directory_path, file_name):
    """
    Save a pandas DataFrame to a CSV file with the given filename
    in the given directory path
    """
    file_path = os.path.join(directory_path, file_name)
    data.to_csv(file_path, index=False)

def correlation_heatmap(data):
    corr = data.corr()
    plt.figure(figsize=(10, 8))
    #plt.figure(figsize=(20, 15))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.show()

    # Select the columns to include in the heatmap
    #columns_to_include = corr_matrix.columns[:]
    #corr_matrix_subset = corr_matrix.loc[columns_to_include, columns_to_include]



def correlation_heatmap(data, title="", xlabel="", ylabel=""):
    corr = data.corr()
    plt.figure(figsize=(20, 15))
    #plt.figure(figsize=(20, 15))
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# This function takes in a dataset and a list of columns to drop from the dataset
def drop_columns(dataset, cols_to_drop):

    # Loop through each column to be dropped
    for col in cols_to_drop:
        # Check if the column exists in the dataset
        if col in dataset.columns:
            # Drop the column from the dataset
            dataset = dataset.drop(columns=col)
    # Return the updated dataset with dropped columns
    return dataset


def min_max_normalization(data):
    """
    Perform Min-Max normalization on a pandas DataFrame
    """
    # Calculate minimum and maximum values for each column
    min_vals = data.min()
    max_vals = data.max()

    # Perform Min-Max normalization on the entire dataset
    data_normalized = (data - min_vals) / (max_vals - min_vals)

    return data_normalized



    import pandas as pd




def perform_pca(data):
    # Load the data from CSV
    df = pd.read_csv(data)

    # Extract the target variable
    y = df.iloc[:, -1]

    # Extract the features
    X = df.iloc[:, :-1]

    # Scale the data
    X_scaled = (X - X.mean()) / X.std()

    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Print the percentage of variance explained by each principal component
    print(pca.explained_variance_ratio_)

    # Return the PCA-transformed data and the target variable
    return X_pca, y




def perform_pca(n_components, data):
    """
    Perform PCA on a given dataset with a specified number of components.

    Parameters:
    - n_components: the number of principal components to retain
    - data: a Pandas dataframe containing the features to transform

    Returns:
    - a new Pandas dataframe containing the principal components
    """

    # Extract the features
    X = data.iloc[:, :-1]

    # Standardize the features by scaling them to have zero mean and unit variance
    X_std = (X - X.mean()) / X.std()

    # Create a PCA object with the specified number of components
    pca = PCA(n_components=n_components)

    # Fit and transform the data
    X_pca = pca.fit_transform(X_std)

    # Create a new dataframe to store the principal components
    columns = [f"PC{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, columns=columns)

    return df_pca








# Define a function to check for null values in a pandas dataframe
def check_null_values(df):
    # Calculate the number of null values for each column in the dataframe
    null_values = df.isnull().sum()
    
    # Print the column name and the number of null values (if any)
    for column, count in null_values.items():
        if count > 0:
            print(f"{column} contains {count} null values.")



def select_random_rows(df, num_rows):
    # Check if the number of rows to select is less than or equal to the number of rows in the dataset
    total_rows = len(df.index)
    if num_rows > total_rows:
        raise ValueError("Number of rows to select cannot be greater than the total number of rows in the dataset.")
    
    # Create a set of indices representing the rows already present in the dataframe
    existing_indices = set(df.index)

    # Select the specified number of random row indices from the dataframe, excluding existing indices
    random_indices = []
    while len(random_indices) < num_rows:
        rand_idx = random.randint(0, total_rows - 1)
        if rand_idx not in existing_indices:
            random_indices.append(rand_idx)

    # Convert the dataframe to a CSV string
    csv_string = df.to_csv(index=False)

    # Load the randomly selected rows into a new dataframe
    random_rows = StringIO()
    for i, row in enumerate(csv_string.split('\n')):
        if i == 0 or i-1 in random_indices:
            random_rows.write(row + '\n')
    random_rows.seek(0)
    random_df = pd.read_csv(random_rows)

    # Return the randomly selected dataframe
    return random_df


