from GradientBoosting.models.GradientBoosting import GradientBoosting
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from itertools import product
import numpy as np


def grid_search(X, y, param_grid):
    """
    Perform grid search to find the best hyperparameters for the Gradient Boosting model.

    Parameters:
    - X: Input features (NumPy array or pandas DataFrame).
    - y: Target variable (NumPy array or pandas Series).
    - param_grid: Dictionary of hyperparameters to search, e.g.,
                  {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}.

    Returns:
    - A dictionary containing the best hyperparameters and the corresponding evaluation metric.
    """
    best_params = None
    best_score = float('inf')  # Lower score is better (MSE)

    # Generate all combinations of hyperparameters
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]

    for params in param_combinations:
        # Create and fit the model with the current hyperparameters
        model = GradientBoosting(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth']
        )

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        # Fit the model
        model.fit(X_train, y_train)

        # Evaluate the model on the test set
        preds = model.predict(X_test)  # Use the trained model for predictions
        mse = mean_squared_error(y_test, preds)

        # Update the best parameters if the current score is better
        if mse < best_score:
            best_score = mse
            best_params = params

    return {
        'best_params': best_params,
        'best_score': best_score
    }
