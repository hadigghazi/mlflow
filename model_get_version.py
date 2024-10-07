import mlflow
import json

def list_model_versions(model_name):
    client = mlflow.tracking.MlflowClient()
    # Search for model versions with the correct filter string
    filter_string = f"name='{model_name}'"
    versions = client.search_model_versions(filter_string)
    # Convert the ModelVersion objects to dictionaries manually
    versions_dict = [{
        'name': version.name,
        'version': version.version,
        'description': version.description,
        'status': version.status,
        'creation_timestamp': version.creation_timestamp,
        'last_updated_timestamp': version.last_updated_timestamp
    } for version in versions]
    return versions_dict

def get_model_version(model_name, version_number):
    client = mlflow.tracking.MlflowClient()
    # List all versions of the model
    filter_string = f"name='{model_name}'"
    versions = client.search_model_versions(filter_string)
    # Find the specific version from the list
    for version in versions:
        if version.version == version_number:
            return {
                'name': version.name,
                'version': version.version,
                'description': version.description,
                'status': version.status,
                'creation_timestamp': version.creation_timestamp,
                'last_updated_timestamp': version.last_updated_timestamp
            }
    return None

if __name__ == "__main__":
    model_name = "DecisionTreeClassifier"  # Ensure this matches the registered model name
    try:
        # List versions of the specified model
        versions = list_model_versions(model_name)
        print(f"Model Versions: {json.dumps(versions, indent=2)}")
    
        
    except Exception as e:
        print(f"An error occurred: {e}")
