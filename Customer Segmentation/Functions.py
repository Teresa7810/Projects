from datetime import date
import pandas as pd
import numpy as np
import random
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix

# Function to count the number of inconsistencies
def invalid_rows(data):
    """
    This function identifies and count rows with invalid values in the 'percentage_of_products_bought_promotion' column.

    Parameters:
        data(panda.DataFrame): The dataset containing multiple rows and columns. It is expected to have a column named.
                        'percentage_of_products_bought_promotion' with percentage values that need to be validated.~

    Returns:
        invalid_row_count(int): The number of rows that have invalid 'percentage_of_products_bought_promotion' values (less than 0 or 
                                greater than 1).
    """
    invalid_rows = data[(data['percentage_of_products_bought_promotion'] < 0) | (data['percentage_of_products_bought_promotion'] > 1)]
    invalid_row_count = invalid_rows.shape[0]
    return invalid_row_count

# Function to find the best number of neighbors
def find_best_k_for_imputation(data, variable_with_missing_values, missing_ratio = 0.1):
    """
    This function determines the optimal number of neighbours (k) for KNN imputation by evaluating the mean squared error
    for different k values on artificially introduced missing data.

     Prameters:
        data(pandas.DataFrame): The dataset containing multiple rows and columns.
        variable_with_missing_values(str): The name of the column with missing values to be imputed.
        missing_ratio(float,optional): The ratio of data points to be randomly set as missing for the purpose of evaluating 
                                    the imputation performance. Default to 0.1 (10%).

    Returns:
        best_k(int): The optimal number of neighbours (k) for KNN imputation.
    """
    complete_data = data[data[variable_with_missing_values].notna()]
    complete_data_copy = complete_data.copy()

    number_missing_values = int(complete_data.shape[0] * missing_ratio)
    missing_indices = random.sample(list(complete_data.index), number_missing_values)
    complete_data_copy.loc[missing_indices, variable_with_missing_values] = np.nan

    original_values = complete_data.loc[missing_indices, variable_with_missing_values]
    
    mean_square_error_scores = []
    
    for k in range(1, 11):
        imputer = KNNImputer(n_neighbors = k)
        imputed_data = imputer.fit_transform(complete_data_copy[[variable_with_missing_values]])
        imputed_values = imputed_data[np.isin(complete_data_copy.index, missing_indices)]

        mean_square_error = mean_squared_error(original_values, imputed_values)
        mean_square_error_scores.append(mean_square_error)

    best_k = range(1, 11)[np.argmin(mean_square_error_scores)]

    return best_k

# Function to calculate the age of a customer
def calculate_age(birthdate):
    """
    This function calculates the age of an individual given their birthdate.

    Parameters:
        birthdate(datetime.data): The birthdate of the individual.
    
    Returns:
        age(int): The age of the individual in years.
    """
    today = date.today()
    age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
    return age

# Function to count the amount of customers per selected variable
def counts(data, column):
    """
    This function calculates the frequency counts of unique values in a specified column of a DataFrame and returns
    the counts sorted by index.

    Parameters:
        data(pandas.DataFrame): The dataset containing multiple rows and columns.
        column(str): The name of the column for which to calculate value counts.

    Returns:
        panda.Series: A series containing the counts of unique values in the specified column, sorted by the index.
    """
    return data[column].value_counts().sort_index()

# Function to compute the total money spent by category
def total_money_spent_by_category(data):
    """
    This function calculates the total money spent for each category based on columns that start with 'spend_'.

    Parameters:
        data(pandas.DataFrame): The dataset containing multiple rows and columns. Columns representing spending
                        should have names starting with 'spend_'.

    Returns:
        total_spending(pandas.DataFrame): A DataFrame with categories as the index and the total money spent in each category as the values.
                                The DataFrame has two columns: 'Category' and 'Total Money Spent'.
    """
    spend_columns = data.columns[data.columns.str.startswith('spend_')]
    total_spending = pd.DataFrame(columns =['Category', 'Total Money Spent'])
    
    for column in spend_columns:
        total_spending = total_spending.append({'Category': column, 'Total Money Spent': data[column].sum()}, ignore_index = True)
    total_spending.set_index('Category', inplace = True)
    return total_spending

# Function to create a buying preferences dataframe
def buying_preferences(data):
    """                          
    This function determines the buying preferences of customers based on their spending on fish and meat.

    Parameters:
        data(pandas.DataFrame): The dataset containing multiple rows and columns. It must include 'spend_fish' and 'spend_meat' columns
                        representing the spending on fish and meat, respectively.

    Returns:
        buying_preferences(pandas.DataFrame): A DataFrame with the buying preference groups as the index and the counts of customers in 
                                    each group as the values. The DataFrame has two columns: 'group' and 'count'
    """
    fish_only = data[(data['spend_fish'] > 0) & (data['spend_meat'] == 0)]
    meat_only = data[(data['spend_fish'] == 0) & (data['spend_meat'] > 0)]
    both_fish_and_meat = data[(data['spend_fish'] > 0) & (data['spend_meat'] > 0)]
    neither_fish_nor_meat = data[(data['spend_fish'] == 0) & (data['spend_meat'] == 0)]
    
    fish_only_count = len(fish_only)
    meat_only_count = len(meat_only)
    both_count = len(both_fish_and_meat)
    neither_count = len(neither_fish_nor_meat)

    buying_preferences = pd.DataFrame({'group': ['Fish Only', 'Meat Only', 'Both Fish and Meat', 'Neither Fish nor Meat'], 'count': [fish_only_count, meat_only_count, both_count, neither_count]})
    buying_preferences.set_index('group', inplace = True)
    
    return buying_preferences

# Group by mean
def group_by_mean(data, variable, n_features = 30):
    """
    This function groups the data by a specified variable and calculates the mean for each group, returning the transposed result.

    Parameters:
        data(pandas.DataFrame): The dataset containing multiple rows and columns.
        variable(str): The column name by which to group the data.
        n_features(int, optional): The number of features (columns) to include in the result. Default to 30.

    Returns:
        pandas.DataFrame: A transposed DataFrame of the grouped mean values, with up to `n_features` columns.
    """
    grouped_data = data.groupby(variable).mean()

    result = grouped_data.iloc[:, :n_features + 1].T

    return result

# Function that generates a dictionary with cluster names as keys and their corresponding colors as values.
def create_cluster_colors_dictionary(colors, labels):
    cluster_colors_dictionary = {label: color for label, color in zip(labels, colors)}
    return cluster_colors_dictionary