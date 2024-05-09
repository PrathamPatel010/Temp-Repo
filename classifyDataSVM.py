from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize Support Vector Machine classifiers with different kernel functions
kernel_functions = ['linear', 'poly', 'rbf', 'sigmoid']

for kernel in kernel_functions:
    # Initialize the SVM classifier
    svm_classifier = SVC(kernel=kernel)

    # Train the classifier
    svm_classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = svm_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy with {kernel} kernel:", accuracy)
