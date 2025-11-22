import mlflow
from mlflow.tracking import MlflowClient
import json
import logging
import os

mlflow.set_tracking_uri("http://ec2-52-23-156-179.compute-1.amazonaws.com:5000")
client=MlflowClient()
experiment=client.get_experiment_by_name("Yt_analysis")
experiment_id=experiment.experiment_id
device="cpu"



logger=logging.getLogger("model_registry")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('log_errors.csv')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def get_root_directory():
   try:
      path=os.path.dirname(os.path.abspath(__file__))
      prev_path=os.path.abspath(os.path.join(path,"../../"))
      return prev_path
   except Exception as e:
      logger.log(logging.ERROR,f"Error in getting root directory: {e}")
      raise


def get_model_path():
    try:
        path=get_root_directory()
        model_path=os.path.join(path,"model_details.json")
        logger.log(logging.INFO,"Getting model path successful")
        return model_path
    except Exception as e:
        logger.log(logging.ERROR,"Error in getting model path:")

def get_model_details():
    try:
        model_path=get_model_path()
        model_details=json.load(open(model_path,"r"))
        logger.log(logging.INFO,"Getting model details successful")
        return model_details
    except:
        logger.log(logging.ERROR,"Error in getting model details")
        raise

    

def register_model(model_details):
    try:
        model_uri=model_details['model_uri']
        mv=mlflow.register_model(model_uri=model_uri,name="sentiment_model")
        client.transition_model_version_stage(
            name="sentiment_model",
            version=mv.version,
            stage="Production",
            archive_existing_versions=True
        )
        print("Succesfully registered")
        logger.log(logging.INFO,"Model registered successfully")
    except:
        logger.log(logging.ERROR,"Error in registering model")
        raise

def main():
    try:
        model_details=get_model_details()
        register_model(model_details)
    except Exception as e:
        logger.log(logging.ERROR,f"Error in main function: {e}")
        raise

if __name__=="__main__":
    main()