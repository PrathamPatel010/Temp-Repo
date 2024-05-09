from sklearn.datasets import load_iris
import numpy as np


class Node:
    def __init__(self, feature=None, value=None, results=None, left=None, right=None):
        self.feature = feature  # Index of the feature to split on
        self.value = value  # Value of the feature to split on
        self.results = results  # Only for leaf nodes: class label or distribution
        self.left = left  # Left subtree
        self.right = right  # Right subtree


def entropy(data1):
    """
    Calculate the entropy of a dataset.
    """
    labels = data1[:, -1]
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropy1 = -np.sum(probabilities * np.log2(probabilities))
    return entropy1


def split_data(data1, feature, value):
    """
    Split the dataset based on the feature and value.
    """
    left = np.array([row for row in data1 if row[feature] <= value])
    right = np.array([row for row in data1 if row[feature] > value])
    return left, right


def find_best_split(data1):
    """
    Find the best feature and value to split the dataset on, based on information gain.
    """
    best_entropy = float('inf')
    best_feature = None
    best_value = None
    num_features = len(data1[0]) - 1  # Last column is the class label

    # Iterate over each feature
    for feature in range(num_features):
        values = np.unique(data1[:, feature])
        # Iterate over unique values of the feature
        for value in values:
            left, right = split_data(data1, feature, value)
            # Skip if the split does not divide the data
            if len(left) == 0 or len(right) == 0:
                continue
            # Calculate information gain
            total_entropy = entropy(data1)
            weighted_entropy = (len(left) / len(data1)) * entropy(left) + (len(right) / len(data1)) * entropy(right)
            information_gain = total_entropy - weighted_entropy
            # Update best split if this split has higher information gain
            if information_gain < best_entropy:
                best_entropy = information_gain
                best_feature = feature
                best_value = value

    return best_feature, best_value


def build_tree(data1):
    """
    Recursively build the decision tree using ID3 algorithm.
    """
    # Base case: if all data points belong to the same class
    if len(np.unique(data1[:, -1])) == 1:
        return Node(results=np.unique(data1[:, -1])[0])

    # Find the best feature and value to split on
    best_feature, best_value = find_best_split(data1)

    # Create a new node with the best split
    left_data, right_data = split_data(data1, best_feature, best_value)
    left_subtree = build_tree(left_data)
    right_subtree = build_tree(right_data)
    return Node(feature=best_feature, value=best_value, left=left_subtree, right=right_subtree)


def classify(tree1, sample1):
    """
    Classify a new sample using the decision tree.
    """
    if tree1.results is not None:
        return tree1.results

    if sample1[tree1.feature] <= tree1.value:
        return classify(tree1.left, sample1)
    else:
        return classify(tree1.right, sample1)


# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
data = np.column_stack((X, y))  # Combine features and target into one array

# Build the decision tree
tree = build_tree(data)

sample = [6.9, 3.1, 5.4, 2.1]
prediction = classify(tree, sample)
print("Classification for sample {}: Class {}".format(sample, prediction))
