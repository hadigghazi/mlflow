import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Student_performance")

df = pd.read_csv('./Student_performance_data _.csv')
df.drop('StudentID', axis=1, inplace=True)

X = df.drop('GradeClass', axis=1)
y = df['GradeClass']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    print("Accuracy:", accuracy)
    print("\nClassification Report:\n", classification_rep)
