import numpy as np
import pandas as pd
from collections import (Counter)
from .DataPreprocessing import (Dataset)
import graphviz

class Node:
    def __init__(self, feature=None, information_gain=None, threshold=None, children=None, *, value=None, correct_cases=None, total_cases=None) -> None:
        # Feature and Threshold this node was divided with
        self.feature = feature
        self.threshold = threshold
        self.information_gain = information_gain
        
        # Defining the Children
        self.children = [] if children is None else children

        # Value of a Node -> Determines if it is a Node or not
        self.value = value
        self.correct_cases = correct_cases
        self.total_cases = total_cases
        self.class_proba = None

    def is_leaf(self) -> bool:
        # If a Node does not have a Value then it is not a Leaf
        return self.value is not None

    def calculate_class_proba(self, y_partition:np.ndarray, target_values:np.ndarray) -> list[float]:
        # Calculates the probability of each target value being in the current y partition
        if (self.total_cases > 0):
            self.class_proba = [(sum(y_partition == target_value) / self.total_cases) for target_value in target_values]
        else:
            self.class_proba = [0 for _ in range(len(target_values))]
    
class DecisionTree:
    def __init__(self, min_samples_split=5, max_depth=10, n_features=None) -> None:
        # Amount of Samples needed to perform a split
        self.min_samples_split = min_samples_split

        # Max depth of the decision tree
        self.max_depth = max_depth

        # Number of features (X) - Helps add some randomness to the Tree
        self.n_features = n_features

        # Defining a root - will later help to traverse the tree
        self.root = None

        # List to store the initial unique values of all the Features
        self.feature_thresholds = None

        # Variable to store the Unique Values of the Target classes
        self.target_values = None

    def _most_common_label(self, y:np.ndarray) -> any:
        # Creating a Counter
        counter = Counter(y)
        
        # Getting the Most Common Value
        value = counter.most_common(1)[0][0]

        # Returns most common value
        return value

    def _entropy(self, y:np.ndarray) -> float:
        # The Bincount method creates a numpy array with the occurences of each value.
        # The index of the array is the number and it's value in the array corresponds to the amount of times it appears in y
        occurences = np.bincount(y)

        # Calculating every pi for every X in the previous array
        ps = occurences / len(y)

        # Returning the Entropy Value
        return - sum(p * np.log2(p) for p in ps if p > 0)

    def _split(self, Attribute_Column:np.ndarray, Thresholds:np.ndarray) -> np.ndarray:
        # Splits the data based on a column's thresholds
        entries_indices = [np.argwhere(Attribute_Column == value).flatten() for value in Thresholds]
        return entries_indices

    def _information_gain(self, y:np.ndarray, Attribute_Column:np.ndarray, Thresholds:np.ndarray) -> float:
        # Calculate Parent Entropy
        parent_entropy = self._entropy(y)

        # Getting unique values in the Column
        unique_thresholds = self._split(Attribute_Column, Thresholds)

        # Calculating the entropies for all the potential children (and the size for each partition)
        children_entropies = [(y[threshold_idx].size, self._entropy(y[threshold_idx])) for threshold_idx in unique_thresholds]

        # Size of the original set
        n = len(y)

        # Initializing the summation to subtract to the parent_entropy in order to obtain the information gain
        sum_child_entropies = 0
        for partition_size, child_entropy in children_entropies:
            sum_child_entropies += ((partition_size / n) * child_entropy)

        return (parent_entropy - sum_child_entropies)

    def _gain_ratio(self, y:np.ndarray, Attribute_Column:np.ndarray, Thresholds:np.ndarray) -> float:
        # Calculates the Gain Ratio
        info_gain = self._information_gain(y, Attribute_Column, Thresholds)
        n = len(y)
        intrinsic_information = 0
        unique_thresholds = self._split(Attribute_Column, Thresholds)

        for unique_threshold in unique_thresholds:
            prop = len(unique_threshold)/n
            intrinsic_information += (prop) * np.log2(prop)
        
        return info_gain / (-intrinsic_information)
        
    def _best_split(self, X:np.ndarray, y:np.ndarray, feature_indices:np.ndarray, method='information_gain') -> tuple[int, float]:
        # Finds the Best Split within the given data partition
        best_information_gain = -1
        best_feature_idx = None

        for feat_idx in feature_indices:
            attribute_column = X[:, feat_idx]
            thresholds = np.unique(attribute_column)

            if (method == 'information_gain'):
                info_gain = self._information_gain(y, attribute_column, thresholds)
            elif (method == 'gain_ratio'):
                info_gain = self._gain_ratio(y, attribute_column, thresholds)
            else:
                raise ValueError('Invalid method selected!!!')
            
            if (info_gain > best_information_gain):
                best_information_gain = info_gain
                best_feature_idx = feat_idx

        return best_feature_idx, best_information_gain

    def _grow_tree(self, X:np.ndarray, y:np.ndarray, feature_indices:np.ndarray, threshold=None, depth=0, method='information_gain') -> Node:
        n_samples, n_features = X.shape
        n_labels = np.unique(y).size
        
        feature_idx, info_gain = self._best_split(X, y, feature_indices, method=method)

        # Base Case
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split or info_gain <= 0):
            most_common_label = self._most_common_label(y)
            new_node = Node(threshold=threshold, value=most_common_label, correct_cases=sum(y == most_common_label), total_cases=len(y))
            new_node.calculate_class_proba(y, self.target_values)
            return new_node
        
        # feature_thresholds = np.unique(X[:, feature_idx])
        feature_thresholds = self.feature_thresholds[feature_idx]

        # Creating the following children partitions
        children = []
        for value in feature_thresholds:
            subset_indices = np.where(X[:, feature_idx] == value)[0]
            
            subset_X = X[subset_indices, :]
            subset_y = y[subset_indices]

            if (len(subset_indices) > 0):
                subtree = self._grow_tree(subset_X, subset_y, np.delete(feature_indices, np.where(feature_indices == feature_idx)), value, depth + 1, method)
                children.append(subtree)
            else:
                node = Node(threshold=value, value=self._most_common_label(y), correct_cases=0, total_cases=0)
                node.calculate_class_proba(y, self.target_values)
                children.append(node)

        return Node(feature_idx, info_gain, threshold, children, total_cases=len(y))

    def fit(self, X:np.ndarray, y:np.ndarray, method='information_gain') -> None:
        # Making sure that the amount of features does not surpass the ones available
        if not self.n_features:
            self.n_features = X.shape[1]
        else:
            self.n_features = min(X.shape[1], self.n_features)

        # Getting the Unique Values of the Features
        self.feature_thresholds = [np.unique(X[:, idx]) for idx in range(self.n_features)]

        # Storing the Target Values
        self.target_values = np.unique(y)
        
        # Creating a Tree Recursively
        self.root = self._grow_tree(X, y, np.arange(self.n_features), method)
    
    def _traverse_tree(self, X:np.ndarray, node:Node) -> any:
        # Traverses the Tree until we reached a leaf node -> which will determine the classification label
        if (node.is_leaf()):
            return node.value

        feature_value = X[node.feature]
        
        for child in node.children:
            if (type(child.threshold) == pd._libs.interval.Interval and (feature_value in child.threshold or feature_value == child.threshold)):
                return self._traverse_tree(X, child)
            elif (str(feature_value) == str(child.threshold)):
                return self._traverse_tree(X, child)

    def predict(self, X:np.ndarray) -> np.ndarray:
        # Predicts the Label given an Input
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree_proba(self, X:np.ndarray, node:Node) -> list[float]:
        # Traverses the Tree until we reached a leaf node -> which will determine the classification label
        if (node.is_leaf()):
            return node.class_proba

        feature_value = X[node.feature]
        
        for child in node.children:
            if (type(child.threshold) == pd._libs.interval.Interval and (feature_value in child.threshold or feature_value == child.threshold)):
                return self._traverse_tree_proba(X, child)
            elif (str(feature_value) == str(child.threshold)):
                return self._traverse_tree_proba(X, child)
    
    def predict_proba(self, X_Test:np.ndarray) -> np.ndarray:
        # Predicts the probability of belonging to each target class
        return np.array([self._traverse_tree_proba(x, self.root) for x in X_Test])
    
    def score(self, y_pred:np.ndarray, y_true:np.ndarray) -> float:
        # Simple Accuracy
        return sum(y_pred == y_true) / len(y_true)

    def print_tree(self, ds:Dataset, node=None, indent=" ") -> None:
        if node is None:
            node = self.root
            if node is None:
                raise Exception("[Unfit Model]")

        # Check if we have a leaf node
        if node.is_leaf():
            print(indent + f"Leaf: Class {ds.target_encoder.inverse_transform([node.value])[0]} [{node.correct_cases}/{node.total_cases}]")
        else:
            # Print current node's feature and information gain
            print(indent + f"Node: {ds.cols[node.feature]} (IG: {node.information_gain:.2f})")
            
            # Iterate over all children of the current node
            for child in node.children:
                # Print the branch and the threshold value
                print(indent + f"Branch: {ds.cols[node.feature]} == {child.threshold}")
                # Recursive call to print each subtree
                self.print_tree(ds, child, indent + "  ")

    def plot_tree(self, ds:Dataset, node:Node, parent_name:str, graph:graphviz.Digraph, counter:int, decision=None) -> int:
        # Base case: leaf node
        if node.is_leaf():
            leaf_name = f'leaf_{counter}'
            graph.node(leaf_name, label=f"{ds.target_encoder.inverse_transform([node.value])[0]} [{node.correct_cases}/{node.total_cases}]", shape='box')
            graph.edge(parent_name, leaf_name, label=decision)
            return counter + 1

        # Displaying Case Analysis in the current node
        feature = ds.cols[node.feature]
        attribute_name = f"{feature} [{node.total_cases}]"
        
        # Adding the Labels to the nodes and braches
        internal_name = f'internal_{counter}'
        graph.node(internal_name, label=attribute_name)
        if parent_name is not None:
            graph.edge(parent_name, internal_name, label=decision)

        counter += 1
        
        for child in node.children:
            label = f'{child.threshold}'
            counter = self.plot_tree(ds, child, internal_name, graph, counter, label)

        return counter

    def visualize_tree(self, dataset:Dataset, file_path=None) -> graphviz.Digraph:
        graph = graphviz.Digraph(format='png', node_attr={'color': 'lightblue2', 'style': 'filled'})
        self.plot_tree(dataset, self.root, None, graph, 0)
        if (file_path is not None):
            # Saves the graph into a .dot file
            graph.save(file_path + '.dot')
            
            # Saves the graph as previously discriminates - as a png
            graph.render(file_path, cleanup=True)
        return graph