## Team Members(Team Falcon):
1. Badri Adusumalli A20530163
2. Bhuvana Chandrika Natharga A20553587
3. Santhosh Kumar Kathiresan A20546185
4. Sriram Ravichandran A20583347




## How to Run the Code

Follow the steps below to set up and run the code on any system. These instructions will guide you through downloading the repository, installing dependencies, and executing the tests.

### Step 1: Download the Repository

1. First, download the repository from GitHub to your local machine. You can do this by either:
   - Cloning the repository using `git clone` command (recommended):
     ```bash
     git clone https://github.com/your-username/your-repo-name.git
     ```
     Replace `your-username/your-repo-name` with the actual URL of your GitHub repository.

   - Alternatively, you can download the ZIP file from GitHub and extract it to your desired location.
  
### Step 2: Open Git Bash and Navigate to the Project Folder

1. Open **Git Bash** (or any command line terminal that supports Git) on your computer.
2. Navigate to the directory where the project is located. For example:
   ```bash
   cd ~/videos/project2
   ```
   In this example, we are assuming that the project is located in the `videos/project2` directory. Replace this path with the actual path where you have downloaded the repository.

### Step 3: Install the Required Dependencies

1. To run the project, you need to install the necessary dependencies listed in the `requirements.txt` file.
2. Use the following command to install all the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
   - This command tells `pip` to install all the packages specified in the `requirements.txt` file. Make sure you have **Python** and **pip** installed on your system. If not, you will need to install them first.

### Step 4: Install the Project in "Editable" Mode Using `setup.py`

1. To allow the project to be used in any location, install it in **editable mode**. This will let Python recognize the `elasticnet` module regardless of your current working directory.
2. Run the following command:
   ```bash
   pip install -e .
   ```
   - The `-e` flag stands for "editable," which allows changes to the source code to be reflected immediately without having to reinstall the package.
   - The `.` specifies the current directory, where the `setup.py` file is located.

### Step 5: Run the Tests to Verify the Installation

1. Now that the dependencies are installed and the project is set up, you can run the tests to ensure everything is working correctly.
2. Execute the following command to run the test file:
   ```bash
   pytest -s GradientBoosting/tests/test_GradientBoosting.py
   ```
   - The `-s` flag ensures that any `print` statements in the test file are displayed in the terminal.
   - `pytest` will run the test cases defined in `test_GradientBoosting.py` to verify the functionality of your GradientBoosting implementation.

### **Step 6: Interactive Input During Testing**

After running the test command, the script will prompt you to provide necessary inputs for testing the Gradient Boosting model:

1. **Dataset File Path**:  
   - You will see the following prompt in the terminal:  
     ```
     Please enter the path to your dataset file:
     ```

2. **Target Column Name**:  
   - After entering the dataset file path, the script will display:  
     ```
     Enter the target column name:
     ```
   - Enter the name of the target column (e.g., `y`) that you wish to use as the dependent variable for training the model.


### **Overview**

---

This project implements a Gradient Boosting model for regression tasks. Gradient Boosting is an ensemble learning method that builds a sequence of weak learners, typically decision trees, where each new learner focuses on correcting the residual errors of the previous ones. It is a powerful and flexible technique for regression problems, known for its ability to handle complex datasets and achieve high predictive accuracy.

### **Key Features**

---

- **Iterative Residual Correction**: The model improves predictions iteratively by minimizing the residual errors from previous models.
- **Decision Tree Base Learners**: Utilizes decision trees as weak learners, which are combined to form a strong predictive model.
- **Learning Rate Control**: Incorporates a learning rate to manage the contribution of each tree and prevent overfitting.
- **Hyperparameter Optimization**: Supports grid search to tune key hyperparameters such as the number of estimators, learning rate, and maximum tree depth for optimal performance.
- **Robustness and Flexibility**: Handles complex data structures, making it well-suited for various regression tasks, even with non-linear relationships.


### Gradient Boosting Implementation

---



### **1. What does the model you have implemented do, and when should it be used?**

The Gradient Boosting model is designed to solve **regression tasks** by combining multiple weak learners (decision trees). It minimizes the error iteratively by learning from residuals, which are the differences between predicted and actual values in the dataset. Each new tree added to the model tries to correct the errors made by the previous trees.

#### **Use Cases**
- **Non-linear Relationships**: Ideal for datasets where relationships between predictors and the target variable are not linear, making traditional linear models unsuitable.
- **High Dimensional Data**: Handles datasets with many features, even when those features have complex interactions.
- **Predictive Accuracy**: Frequently used in competitions (like Kaggle) due to its ability to provide state-of-the-art results in regression tasks.
- **Robustness**: Suitable for scenarios where overfitting must be controlled through learning rates and regularization.

#### **When to Use It**
- When predictive accuracy is a priority.
- When your dataset exhibits non-linear relationships and interactions between variables.
- When interpretability is less critical (as Gradient Boosting models are complex compared to linear regression).
- When you want a model that performs well out-of-the-box but allows for fine-tuning through hyperparameters.

---

### **2. How did you test your model to determine if it is working reasonably correctly?**

The model's correctness and effectiveness were validated through the following steps:

1. **Test Dataset**:
   - The model was tested on synthetic datasets with known properties to verify its ability to approximate the underlying patterns and minimize residual errors.
   - Example: A generated dataset with non-linear relationships between features and the target variable.

2. **Evaluation Metrics**:
   - **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted values. Lower values indicate better performance.
   - **Root Mean Squared Error (RMSE)**: Provides a more interpretable measure by putting the error on the same scale as the target variable.
   - **R² (Coefficient of Determination)**: Indicates the proportion of variance in the target variable explained by the model.

3. **Visualization**:
   - **Density Plot**: Compares the distribution of actual versus predicted values to assess alignment.
   - **Prediction Error Plot**: Shows how well predictions align with actual values using scatterplots.

4. **Cross-Validation**:
   - During hyperparameter tuning via grid search, the dataset was split into training and testing sets to evaluate generalization and avoid overfitting.

5. **Edge Case Testing**:
   - Tested the model with datasets containing missing values to ensure null handling works correctly.
   - Ensured stability when presented with datasets with correlated features or large variance in feature scales.

---

### **3. What parameters have you exposed to users of your implementation in order to tune performance?**

The implementation allows users to tune the following parameters for performance optimization:

1. **Number of Estimators (`n_estimators`)**:
   - Specifies the number of decision trees in the ensemble.
   - More trees generally improve performance but increase computational cost and risk of overfitting.
   - Example: `n_estimators = 50` or `n_estimators = 150`.

2. **Learning Rate (`learning_rate`)**:
   - Controls the contribution of each tree to the overall prediction.
   - A smaller learning rate requires more trees to achieve the same performance but improves generalization.
   - Example: `learning_rate = 0.05`.

3. **Maximum Depth of Trees (`max_depth`)**:
   - Restricts the depth of each decision tree, controlling its complexity.
   - A deeper tree captures more intricate patterns but increases the risk of overfitting.
   - Example: `max_depth = 3`.

#### **Basic Usage Example**
```python
from GradientBoosting.models.GradientBoosting import GradientBoosting

# Initialize the model
model = GradientBoosting(n_estimators=100, learning_rate=0.1, max_depth=3)

# Fit the model to training data
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

#### **Hyperparameter Tuning**
- Use `grid_search` to automatically find the optimal combination of parameters:
```python
from GradientBoosting.models.grid_search import grid_search

param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

best_params = grid_search(X_train, y_train, param_grid)
```

---

### **4. Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?**

#### **Current Limitations**:
1. **Categorical Features**:
   - The model currently expects numeric inputs and does not support automatic encoding of categorical variables.
   - **Workaround**: Preprocess categorical data using `OneHotEncoder` or similar techniques before passing it to the model.
   - **Future Enhancement**: Integrate categorical feature support directly into the model.

2. **Outliers**:
   - Extreme outliers in the dataset can skew the residuals, affecting the performance of subsequent trees.
   - **Workaround**: Use preprocessing steps such as outlier removal or robust scaling before fitting the model.

3. **Imbalanced Datasets**:
   - The current implementation is not optimized for datasets with highly imbalanced target distributions.
   - **Workaround**: Use techniques like oversampling, undersampling, or appropriate evaluation metrics to address imbalance.

4. **Computational Cost**:
   - The model may become computationally expensive for large datasets or when using a high number of estimators.
   - **Workaround**: Use a smaller learning rate and fewer estimators while monitoring performance. Parallelize tree building if possible.

#### **Future Directions**:
- **Feature Engineering**: Automate feature preprocessing (e.g., handling categorical data and missing values).
- **Early Stopping**: Implement early stopping to halt training when performance ceases to improve on validation data.
- **Explainability**: Add tools to interpret feature importance for better model explainability.




