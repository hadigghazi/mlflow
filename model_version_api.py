from flask import Flask, jsonify, request, abort
import mlflow

app = Flask(__name__)

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
        if str(version.version) == str(version_number):
            return {
                'name': version.name,
                'version': version.version,
                'description': version.description,
                'status': version.status,
                'creation_timestamp': version.creation_timestamp,
                'last_updated_timestamp': version.last_updated_timestamp
            }
    return None

@app.route('/models/<model_name>/versions', methods=['GET'])
def get_versions(model_name):
    try:
        versions = list_model_versions(model_name)
        return jsonify(versions)
    except Exception as e:
        abort(500, description=f"An error occurred: {e}")

@app.route('/models/<model_name>/versions/<version_number>', methods=['GET'])
def get_version(model_name, version_number):
    try:
        version = get_model_version(model_name, version_number)
        if version:
            return jsonify(version)
        else:
            abort(404, description=f"Model version {version_number} not found.")
    except Exception as e:
        abort(500, description=f"An error occurred: {e}")

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
