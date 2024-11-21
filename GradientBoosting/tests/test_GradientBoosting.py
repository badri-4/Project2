import csv
import numpy as np
import pandas as pd
from GradientBoosting.models.GradientBoosting import GradientBoosting
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt


def test_predict():
    # Initialize GradientBoosting model
    # n_estimators: Number of trees in the ensemble
    # learning_rate: Step size for updating residuals
    # max_depth: Maximum depth of each decision tree
    model = GradientBoosting(n_estimators=100, learning_rate=0.1, max_depth=3)

    # Load data from CSV file
    # csv_file_path: Path to the dataset
    csv_file_path = "GradientBoosting/tests/small_test.csv"
    df = pd.read_csv(csv_file_path)  # Load data into a pandas DataFrame

    # Separating features (X) and target variable (y)
    # X contains all columns except 'y', which is the target
    X = df.drop(columns=['y'])
    y = df['y']

    # Handling categorical columns in X
    # Identifies categorical columns and applies OneHotEncoding
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(sparse=False, drop='first')  # OneHotEncoder avoids collinearity
        # Encode categorical columns and add to DataFrame
        X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]), index=X.index)
        X_encoded.columns = encoder.get_feature_names_out(categorical_cols)
        X = X.drop(columns=categorical_cols)  # Drop original categorical columns
        X = pd.concat([X, X_encoded], axis=1)  # Add encoded columns to X

    # Handling categorical target variable y
    # Converts target variable y into numeric using LabelEncoder if categorical
    if y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    # Convert y to NumPy array to avoid index mismatches
    y = np.array(y)

    # Split data into training and testing sets (80-20 split)
    # Training set is 80%, Testing set is 20%
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.20, random_state=42)

    # Fit the GradientBoosting model on training data
    # Model learns the relationship between X_train and y_train
    results = model.fit(X_train, y_train)

    # Predict using the fitted model on the test set
    preds = results.predict(X_test)

    # Calculate evaluation metrics
    # Mean Squared Error (MSE): Measures the average squared difference between predictions and actual values
    mse = mean_squared_error(y_test, preds)
    # R-squared (R²): Proportion of variance in the dependent variable explained by the model
    r2 = r2_score(y_test, preds)

    # Print evaluation metrics
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R²): {r2}")

    # **1. Residual Plot**
    # Residual = Predicted - Actual
    # Visualizes residuals to assess model bias
    residuals = preds - y_test
    plt.scatter(y_test, residuals)
    plt.axhline(y=0, color='r', linestyle='--')  # Reference line at residual=0
    plt.xlabel("True Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.show()

    # **2. Predicted vs. True Values**
    # Visualizes how well predictions align with actual values
    plt.scatter(y_test, preds)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='r', linestyle='--')  # Ideal fit line
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("True vs. Predicted Values")
    plt.show()

    # **3. Learning Curve**
    # Visualizes training and testing errors as the number of trees increases
    train_errors = []
    test_errors = []
    for i, tree in enumerate(model.trees):
        # Calculate predictions for training data using the first i+1 trees
        partial_preds_train = np.sum([tree.predict(X_train) for tree in model.trees[:i+1]], axis=0)
        train_errors.append(mean_squared_error(y_train, partial_preds_train))

        # Calculate predictions for testing data using the first i+1 trees
        partial_preds_test = np.sum([tree.predict(X_test) for tree in model.trees[:i+1]], axis=0)
        test_errors.append(mean_squared_error(y_test, partial_preds_test))

    # Plot the learning curve
    plt.plot(train_errors, label='Training Error')
    plt.plot(test_errors, label='Testing Error')
    plt.xlabel("Number of Trees")
    plt.ylabel("Mean Squared Error")
    plt.title("Learning Curve")
    plt.legend()  # Add a legend to differentiate training and testing errors
    plt.show()

    # Dummy assertion to validate test structure
    # Ensures that predictions are returned as a NumPy array
    assert isinstance(preds, np.ndarray), "Prediction is not an array"


# Run the test function if executed directly
if __name__ == "__main__":
    test_predict()
