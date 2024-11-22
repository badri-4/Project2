import numpy as np


def fill_if_null(data):
    """
    Fill null values in a DataFrame with the mean of each column.

    Parameters:
    - data: pandas DataFrame

    Returns:
    - data: pandas DataFrame with nulls filled
    """
    null_boy = np.array(data.columns[data.isnull().any()])
    for i in null_boy:
        data[i] = data[i].fillna(data[i].mean())
    return data


def check_null(data):
    """
    Check for null values in a DataFrame and fill them if found.

    Parameters:
    - data: pandas DataFrame

    Returns:
    - None: Prints the count of null values in each column.
    """
    if data.isnull().values.any():
        fill_if_null(data)
        print(data.isnull().sum())
    else:
        print(data.isnull().sum())


def XandY(data, target_column):
    """
    Split the DataFrame into features (X) and target (Y).

    Parameters:
    - data: pandas DataFrame
    - target_column: str, name of the target column

    Returns:
    - X: NumPy array of features
    - Y: NumPy array of target
    """
    Y = data[target_column].to_numpy()
    data.drop(target_column, axis=1, inplace=True)
    X = data.to_numpy()

    return [X, Y]
