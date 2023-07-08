import mlflow
import os

print('--------------------------------------------')
root_path = os.getcwd()
# path = root_path + '/app'
run_id = 'd08c5bac354b4c92bcba274b2ccb5247'

mlflow.set_tracking_uri(f'file:./app/mlruns') 

print(root_path)

ticket_settings = mlflow.artifacts.load_dict(
        #artifact_uri=f"file:///app/mlruns/{run_id}/artifacts/settings.json"
        f'runs:/{run_id}/settings.json'
    )
print('artifact donee')

run_id = 'de181724bd1b4a73a30f0c0a00315b97'
proj_name = 'ImageFinder'

model_local_path = mlflow.artifacts.download_artifacts(
        run_id= run_id,
        artifact_path=f"{proj_name}-run"
    )   

model = mlflow.pytorch.load_model(model_local_path)

print('call worked')