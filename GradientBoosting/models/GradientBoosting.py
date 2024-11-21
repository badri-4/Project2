import numpy as np


class DecisionTree:
    def __init__(self, max_depth=3):
        """
        Initialize the DecisionTree with a specified maximum depth.

        Parameters:
        - max_depth: Maximum depth of the decision tree.
        """
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        """
        Fit a decision tree to the given data.

        Parameters:
        - X: Input features (NumPy array).
        - y: Target variable (NumPy array).
        """
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grow the decision tree by splitting nodes.

        Parameters:
        - X: Input features for the current node.
        - y: Target variable for the current node.
        - depth: Current depth of the tree.

        Returns:
        - A dictionary representing the tree structure.
        """
        n_samples, n_features = X.shape

        # Base case: If maximum depth is reached or only one sample remains
        if depth >= self.max_depth or n_samples <= 1:
            # Create a leaf node with the mean value of the target variable
            leaf_value = np.mean(y)
            return {'leaf': leaf_value}

        # Find the best feature and threshold to split the data
        best_split = self._find_best_split(X, y, n_features)

        # If no valid split is found, return a leaf node
        if not best_split:
            leaf_value = np.mean(y)
            return {'leaf': leaf_value}

        # Recursively grow left and right subtrees
        left_indices, right_indices = best_split['left_indices'], best_split['right_indices']
        left_tree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        # Return the split node
        return {'feature': best_split['feature'], 'threshold': best_split['threshold'], 'left': left_tree,
                'right': right_tree}

    def _find_best_split(self, X, y, n_features):
        """
        Find the best feature and threshold to split the data.

        Parameters:
        - X: Input features.
        - y: Target variable.
        - n_features: Number of features.

        Returns:
        - A dictionary containing the best split information, or None if no split is found.
        """
        y = np.array(y)  # Ensure compatibility with NumPy indexing
        best_split = {}
        min_mse = float('inf')  # Start with a very high MSE

        # Iterate over each feature
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])  # Get all unique values for the feature
            for threshold in thresholds:
                # Split data into left and right based on the threshold
                left_indices = np.where(X[:, feature_index] <= threshold)[0]
                right_indices = np.where(X[:, feature_index] > threshold)[0]

                # Skip invalid splits
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                # Calculate mean squared error for the split
                mse = self._calculate_mse(y[left_indices], y[right_indices])
                if mse < min_mse:
                    min_mse = mse
                    best_split = {
                        'feature': feature_index,
                        'threshold': threshold,
                        'left_indices': left_indices,
                        'right_indices': right_indices
                    }
        return best_split if best_split else None

    def _calculate_mse(self, left_y, right_y):
        """
        Calculate the mean squared error for a split.

        Parameters:
        - left_y: Target values for the left split.
        - right_y: Target values for the right split.

        Returns:
        - Mean squared error for the split.
        """
        left_mse = np.var(left_y) * len(left_y)
        right_mse = np.var(right_y) * len(right_y)
        return (left_mse + right_mse) / (len(left_y) + len(right_y))

    def predict(self, X):
        """
        Predict target values using the fitted decision tree.

        Parameters:
        - X: Input features.

        Returns:
        - Predicted target values.
        """
        return np.array([self._predict_sample(sample) for sample in X])

    def _predict_sample(self, sample):
        """
        Predict a single sample by traversing the tree.

        Parameters:
        - sample: A single input sample.

        Returns:
        - Predicted value for the sample.
        """
        node = self.tree
        while 'leaf' not in node:
            # Traverse left or right based on the feature threshold
            if sample[node['feature']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node['leaf']


class GradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        """
        Initialize the GradientBoosting model.

        Parameters:
        - n_estimators: Number of decision trees in the ensemble.
        - learning_rate: Step size for updating residuals.
        - max_depth: Maximum depth of each decision tree.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = 0

    def fit(self, X, y):
        """
        Fit the Gradient Boosting model to the data.

        Parameters:
        - X: Input features.
        - y: Target variable.

        Returns:
        - GradientBoostingResults containing the fitted model.
        """
        # Initialize the first prediction as the mean of the target variable
        self.initial_prediction = np.mean(y)
        residuals = y - self.initial_prediction

        # Train `n_estimators` decision trees
        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)  # Fit tree on current residuals
            predictions = tree.predict(X)  # Get predictions from the tree
            residuals -= self.learning_rate * predictions  # Update residuals
            self.trees.append(tree)  # Store the fitted tree

        return GradientBoostingResults(self.initial_prediction, self.trees, self.learning_rate)


class GradientBoostingResults:
    def __init__(self, initial_prediction, trees, learning_rate):
        """
        Store results of the Gradient Boosting model.

        Parameters:
        - initial_prediction: The initial prediction (mean of the target variable).
        - trees: List of fitted decision trees.
        - learning_rate: Learning rate used for updating residuals.
        """
        self.initial_prediction = initial_prediction
        self.trees = trees
        self.learning_rate = learning_rate

    def predict(self, X):
        """
        Predict using the fitted Gradient Boosting model.

        Parameters:
        - X: Input features.

        Returns:
        - Predicted target values.
        """
        # Start with the initial prediction
        y_pred = np.full(X.shape[0], self.initial_prediction)
        # Add predictions from all trees
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred
