import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from GradientBoosting.models.GradientBoosting import GradientBoosting
from GradientBoosting.models.grid_search import grid_search
from GradientBoosting.models.Check import check_null, XandY


def test_predict():
    """
    Test the GradientBoosting model with a dataset, evaluate its performance, and visualize results.
    """

    #! If you are going to use "pytest", enable this block
    # file_path = "GradientBoosting/tests/small_test.csv"
    # df = pd.read_csv(file_path)
    # target = 'y'

    #! Comment it out if you are using "pytest"
    file_path = input("Please enter the path to your dataset file: ")

    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            print("Unsupported file format. Please provide a CSV, Excel, JSON, or Parquet file.")
            return
    except FileNotFoundError:
        print("File not found. Please check the path and try again.")
        return

    print("\n" + "=" * 40)
    print("Dataset Preview:")
    print("=" * 40)
    print(df.head())

    #! Uncomment this block if using "pytest"
    # target = 'y'

    #! Comment out this block if using "pytest"
    target = input("Enter the target column name: ")

    # Check and handle null values
    check_null(df)

    # Split data into features (X) and target (Y)
    X, Y = XandY(df, target)

    # Split data into training and testing sets
    np.random.seed(42)
    shuffled_indices = np.random.permutation(X.shape[0])
    train_size = int(0.8 * len(shuffled_indices))
    train_indices, test_indices = shuffled_indices[:train_size], shuffled_indices[train_size:]
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = Y[train_indices], Y[test_indices]

    # Define hyperparameters for grid search
    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }

    # Perform grid search to find the best hyperparameters
    grid_results = grid_search(X_train, y_train, param_grid)
    best_params = grid_results['best_params']

    print("\n" + "=" * 40)
    print("Best Parameters from Grid Search")
    print("=" * 40)
    print(f"Number of Estimators: {best_params['n_estimators']}")
    print(f"Learning Rate: {best_params['learning_rate']}")
    print(f"Maximum Depth: {best_params['max_depth']}")
    print(f"Best MSE: {grid_results['best_score']:.4f}")
    print("=" * 40)

    # Initialize the model with the best parameters
    final_model = GradientBoosting(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth']
    )

    # Train the final model
    final_model.fit(X_train, y_train)
    final_predictions = final_model.predict(X_test)

    # Calculate evaluation metrics
    mse = np.mean((y_test - final_predictions) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum((y_test - final_predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

    print("\n" + "=" * 40)
    print("Final Model Evaluation")
    print("=" * 40)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print("=" * 40)

    # Visualization 1: Density Plot of Actual vs Predicted Values
    plt.figure(figsize=(8, 6))
    sns.kdeplot(y_test, color='blue', fill=True, label='Actual Values')
    sns.kdeplot(final_predictions, color='blue', fill=True, label='Predicted Values')
    plt.title('Density Plot of Actual vs Predicted Values')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Visualization 2: Prediction Error Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, final_predictions, color='green', label='Predicted Values', alpha=0.6)
    plt.plot(
        [min(y_test), max(y_test)], [min(y_test), max(y_test)],
        color='red', linestyle='--', label='Perfect Prediction'
    )
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Prediction Error Plot')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    test_predict()
