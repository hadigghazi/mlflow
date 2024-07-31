import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

def get_production_model_version(name):
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    versions = client.get_registered_model(name).latest_versions
    for version in versions:
        if version.current_stage == "Production":
            return version
    raise ValueError(f"No production version found for model: {name}")

def test_model(name, model_version, X_test, y_test):
    model_uri = f"models:/{name}/{model_version.version}"
    model = mlflow.pyfunc.load_model(model_uri)
    y_pred = model.predict(X_test)
    return {"accuracy": accuracy_score(y_test, y_pred)}

# Setting up MLflow tracking
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Student_performance")

# Load data
df = pd.read_csv('./Student_performance_data _.csv')
df.drop('StudentID', axis=1, inplace=True)
X = df.drop('GradeClass', axis=1)
y = df['GradeClass']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training and logging model
with mlflow.start_run():
    mlflow.set_tag("developer", "hady")
    mlflow.set_tag("model", "decision-tree")
    
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)
    
    mlflow.sklearn.log_model(clf, "model")
    
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    mlflow.log_dict(classification_rep, "classification_report.json")

# Get the production model version
production_version = get_production_model_version(name="Student_performance")

# Test the production model version
test_results = test_model(name="Student_performance", model_version=production_version, X_test=X_test, y_test=y_test)
print(test_results)

# Print the production model version and stage
print(f"Production version: {production_version.version}, stage: {production_version.current_stage}")

# Search for specific runs with accuracy above a threshold
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
runs = client.search_runs(
    experiment_ids='1',
    filter_string="metrics.accuracy >0.82",
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=5,
    order_by=["metrics.accuracy DESC"]
)
