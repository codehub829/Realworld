# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data  # Features (sepal length, sepal width, petal length, petal width)
y = iris.target  # Labels (flower species)

# Step 2: Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier()

# Step 4: Train the classifier with the training data
clf.fit(X_train, y_train)

# Step 5: Make predictions on the test data
y_pred = clf.predict(X_test)

# Step 6: Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Step 7: Print the results
print(f"Model Accuracy: {accuracy * 100:.2f}%")
