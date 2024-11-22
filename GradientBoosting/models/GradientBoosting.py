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

        if depth >= self.max_depth or n_samples <= 1:
            leaf_value = np.mean(y)
            return {'leaf': leaf_value}

        best_split = self._find_best_split(X, y, n_features)

        if not best_split:
            leaf_value = np.mean(y)
            return {'leaf': leaf_value}

        left_indices, right_indices = best_split['left_indices'], best_split['right_indices']
        left_tree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return {
            'feature': best_split['feature'],
            'threshold': best_split['threshold'],
            'left': left_tree,
            'right': right_tree,
        }

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
        best_split = {}
        min_mse = float('inf')

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = np.where(X[:, feature_index] <= threshold)[0]
                right_indices = np.where(X[:, feature_index] > threshold)[0]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                mse = self._calculate_mse(y[left_indices], y[right_indices])
                if mse < min_mse:
                    min_mse = mse
                    best_split = {
                        'feature': feature_index,
                        'threshold': threshold,
                        'left_indices': left_indices,
                        'right_indices': right_indices,
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
        """
        self.initial_prediction = np.mean(y)
        residuals = y - self.initial_prediction

        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            predictions = tree.predict(X)
            residuals -= self.learning_rate * predictions
            self.trees.append(tree)

    def predict(self, X):
        """
        Predict using the fitted Gradient Boosting model.

        Parameters:
        - X: Input features.

        Returns:
        - Predicted target values as a NumPy array.
        """
        y_pred = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred
