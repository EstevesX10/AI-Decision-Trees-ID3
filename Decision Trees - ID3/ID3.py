import numpy as np
import pandas as pd
from collections import (Counter)

class Node:
    def __init__(self, feature=None, threshold=None, information_gain=None, left=None, right=None, *, value=None):
        # Feature and Threshold this node was divided with
        self.feature = feature
        self.threshold = threshold
        self.information_gain = information_gain
        
        # Defining the Left and Right children
        self.left = left
        self.right = right

        # Value of a Node -> Determines if it is a Node or not
        self.value = value

    def is_leaf(self):
        # If a Node does not have a Value then it is not a Leaf
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        # Amount of Samples needed to perform a split
        self.min_samples_split = min_samples_split

        # Max depth of the decision tree
        self.max_depth = max_depth

        # Number of features (X) - Helps add some randomness to the Tree
        self.n_features = n_features

        # Defining a root - will later help to traverse the tree
        self.root = None

    def _most_common_label(self, y):
        # Creating a Counter
        counter = Counter(y)
        
        # Getting the Most Common Value
        value = counter.most_common(1)[0][0]

        # Returns most common value
        return value

    def _entropy(self, y):
        # The Bincount method creates a numpy array with the occurences of each value.
        # The index of the array is the number and it's value in the array corresponds to the amount of times it appears in y
        occurences = np.bincount(y)

        # Calculating every pi for every X in the previous array
        ps = occurences / len(y)

        # Returning the Entropy Value
        return - sum(p * np.log2(p) for p in ps if p > 0)

    def _split(self, X_Column, split_threshold):
        # Splitting the Data
        # Note: np.argwhere().flatten() returs the list of indices from the given one where it's elements obey the condition given
        left_indices = np.argwhere(X_Column <= split_threshold).flatten()
        right_indices = np.argwhere(X_Column > split_threshold).flatten()
        return left_indices, right_indices

    def _information_gain(self, y, X_Column, threshold):
        # Getting the Parent Entropy
        parent_entropy = self._entropy(y)

        # Create the Children
        left_indices, right_indices = self._split(X_Column, threshold)

        # Checks if any of the lists are empty
        if (left_indices.size == 0 or right_indices.size == 0):
            return 0

        # -> Calculate the Weighted Average Entropy of the Children

        # Number of Samples in y
        n = len(y)

        # Number of samples in the Left and Right children
        n_left, n_right = left_indices.size, right_indices.size

        # Calculate the Entropy for both Samples (Left and Right)
        entropy_left, entropy_right = self._entropy(y[left_indices]), self._entropy(y[right_indices])

        # Calculate the Child Entropy
        child_entropy = (n_left / n) * entropy_left + (n_right / n) * entropy_right

        # Calculate Information Gain
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _best_split(self, X, y, feature_indices):
        # Finds the Best existent split and threshold (Based on the Information Gain)

        # Initializing the Best Parameters
        best_gain = -1
        split_idx, split_threshold = None, None

        # Traverse all possible actions
        for feat_idx in feature_indices:
            X_Column = X[:, feat_idx]
            thresholds = np.unique(X_Column)

            for threshold in thresholds[:-1]:
                # Calculate the Information Gain
                gain = self._information_gain(y, X_Column, threshold)

                # Updating the Best Parameters
                if (gain > best_gain):
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = threshold

        # Returning the Best Split Criteria Found
        return split_idx, split_threshold, best_gain

    def _grow_tree(self, X, y, depth=0):
        # Getting the number of samples, features and labels in the data given
        n_samples, n_features = X.shape
        n_labels = np.unique(y).size
        
        """
        # Stopping Criteria

        (depth >= self.max_depth)             => Reached Maximum depth defined
        (n_labels == 1)                       => Current Node only has 1 type of label (which means it's pure)
        (n_samples < self.min_samples_split)  => The amount of samples is not enough to perform a split

        Therefore, we must return a new node (which is going to be a leaf)
        with the current inform
        """

        # Checks the Stopping Criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Getting the Indices of the Features
        features_indices = np.random.choice(n_features, self.n_features, replace=False)

        # Find the Best Split 
        best_feature, best_threshold, info_gain = self._best_split(X, y, features_indices)

        # Create Child Nodes (Also makes a recursive call to continue to grow the tree)
        left_indices, right_indices = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_indices, :], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices, :], y[right_indices], depth + 1)
        
        return Node(best_feature, best_threshold, info_gain, left, right)

    def fit(self, X, y):
        # Making sure that the amount of features does not surpass the ones available
        if not self.n_features:
            self.n_features = X.shape[1]
        else:
            self.n_features = min(X.shape[1], self.n_features)
    
        # Creating a Tree Recursively
        self.root = self._grow_tree(X, y)
            
    def _traverse_tree(self, X, node:Node):
        # Traverses the Tree until we reached a leaf node -> which will determine the classification label
        if (node.is_leaf()):
            return node.value

        if (X[node.feature] <= node.threshold):
            return self._traverse_tree(X, node.left)
        else:
            return self._traverse_tree(X, node.right)

    def predict(self, X):
        # Predicts the Label given an Input
        return np.array([self._traverse_tree(x, self.root) for x in X])