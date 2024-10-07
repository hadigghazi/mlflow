import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Train a Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Start an MLflow run
with mlflow.start_run() as run:
    # Log the model
    mlflow.sklearn.log_model(model, "model")

    # Register the model with a specific name
    mlflow.register_model("runs:/" + run.info.run_id + "/model", "DecisionTreeClassifier")

    # Log parameters and metrics
    mlflow.log_param("criterion", model.criterion)
    mlflow.log_param("splitter", model.splitter)
    mlflow.log_metric("accuracy", model.score(X_test, y_test))

    # Visualize the tree using matplotlib
    plt.figure(figsize=(20,10))
    plot_tree(model, feature_names=data.feature_names, class_names=data.target_names, filled=True)
    plt.title('Decision Tree Visualization')
    plt.show()
    plt.close()  # Close the plot to avoid displaying it

print(f"Model versioned in run: {run.info.run_id}")
