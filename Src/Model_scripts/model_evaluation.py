import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import dataset,DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import yaml
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
import mlflow
from mlflow.tracking import MlflowClient
import json

mlflow.set_tracking_uri("http://ec2-52-23-156-179.compute-1.amazonaws.com:5000")
client=MlflowClient()
experiment=client.get_experiment_by_name("Yt_analysis")
experiment_id=experiment.experiment_id
device="cpu"


logger=logging.getLogger("model_evaluation")
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

class Lstm_model(nn.Module):
  def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim=3):
    super().__init__()
    self.embedding=nn.Embedding(vocab_size,embedding_dim)
    self.lstm=nn.LSTM(embedding_dim,hidden_dim,batch_first=True)
    self.linear=nn.Linear(hidden_dim,output_dim)

  def forward(self,x):
    x=self.embedding(x)
    output,hidden=self.lstm(x)
    x=hidden[-1]
    y=self.linear(x)
    return y

def load_test_data(test_path):
    try:
        test_df=pd.read_csv(test_path)
        logger.log(logging.INFO, f"Test data loaded successfully from {test_path}")
        return test_df
    except Exception as e:
        logger.log(logging.ERROR, f"An error occurred while loading test data: {e}")
        raise
    

def load_model(model_config):
    try:
        model_config=json.load(open(model_config,'r'))
        model_path=model_config['model_path']
        vocab_size=model_config['vocab_size']
        embedding_dim=model_config['embedding_dim'] 
        hidden_dim=model_config['hidden_dim']
        model=Lstm_model(vocab_size,embedding_dim=embedding_dim,hidden_dim=hidden_dim)
        model_wts=torch.load(model_path)
        model.load_state_dict(model_wts)
        logger.log(logging.INFO, f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.log(logging.ERROR, f"An error occurred while loading the model: {e}")
        raise

def get_tokenizer(df,type="test"):
    try:
        tokenizer=Tokenizer(num_words=2000,lower=True)
        tokenizer.fit_on_texts(df['processed_text'])
        logger.log(logging.INFO,"Tokenizer created successfully")
        return tokenizer
    except Exception as e:
       logger.log(logging.ERROR,f"An error occured while getting tokenizer: {e}")
       raise


def get_root_directory():
   try:
      path=os.path.dirname(os.path.abspath(__file__))
      prev_path=os.path.abspath(os.path.join(path,"../../"))
      return prev_path
   except Exception as e:
      logger.log(logging.ERROR,f"Error in getting root directory: {e}")
      raise





def text_to_vec(df,type="test",max_len=900):
    try:
        tokenizer=get_tokenizer(df)
        try:
            X=tokenizer.texts_to_sequences(df['new_clean_comment'])
        except:
            logger.log(logging.ERROR,"Error in converting text to sequence")
            raise
        try:
            X=pad_sequences(X,maxlen=max_len)
        except Exception as e:
            logger.log(logging.ERROR,f"Error in padding sequences:{e}")
            raise
        y=np.array(df['category'])
        if type=="train":
            vocab=tokenizer.word_index
            return X,y,len(vocab)
        return X,y
    except Exception as e:
        logger.log(logging.ERROR,f"An error occurred while converting text to vector: {e}")
        raise


def get_train_test_dataloader(test_df,batch_size=4):
    try:
        X_test,y_test=text_to_vec(test_df)
        X_test=torch.tensor(X_test,dtype=torch.long).to(device)
        y_test= torch.tensor(y_test,dtype=torch.long).to(device)
        test_dataset=TensorDataset(X_test,y_test)
        batch_size=4
        test_dataloader=DataLoader(test_dataset,batch_size=batch_size,shuffle=True)
        logger.log(logging.INFO,"Test data loaders created successfully")
        return test_dataloader
    except Exception as e:
        logger.log(logging.ERROR,f"An error occurred while creating train and test dataloaders: {e}")
        raise

def evaluate_model(model,test_dataloader):
  model.eval()
  y_pred_all=[]
  y_true_all=[]
  for x,y in test_dataloader:
    y_pred=model(x)
    y_pred=torch.max(y_pred,-1).indices.view(y.shape[0])
    if y_pred.device.type=="cuda":
      y_pred=y_pred.to("cpu").tolist()
      y=y.to("cpu").tolist()
      y_pred_all.extend(y_pred)
      y_true_all.extend(y)
    elif y_pred.device.type=="cpu":
      y_pred=y_pred.tolist()
      y_pred_all.extend(y_pred)
      y=y.tolist()
      y_true_all.extend(y)
  report=classification_report(y_true_all,y_pred_all,output_dict=True)
  return report,y_true_all,y_pred_all


def store_best_model_lstm(model,test_dataloader):
    run = client.create_run(experiment_id=experiment_id)
    run_id = run.info.run_id
    class_report,y_true_all,y_pred_all=evaluate_model(model,test_dataloader)
    print("class report:",class_report)
    client.log_param(run_id,"Model","LSTM")
    for metric in class_report:
        if type(class_report[metric])==dict:
            for key in class_report[metric]:
                if key!='support':
                    client.log_metric(run_id,f"{metric}_{key}",class_report[metric][key])
        else:
            client.log_metric(run_id,metric,class_report[metric])
    model_path=mlflow.pytorch.log_model(model, "sentiment_model")
    model_path=model_path.model_uri
    path=get_root_directory()
    with open(path+"/model_details.json","w") as f:
        details={
            "run_id":run_id,
            "model_uri":model_path
        }
        f.write(json.dumps(details))
        logger.log(logging.INFO,"Model details saved successfully")


def main():
    try:
        path=get_root_directory()
        test_path=os.path.join(path,"Data/Clean/test_clean.csv")
        model_config=path+"/model_config.json"
        model=load_model(model_config)
        test_df=load_test_data(test_path)
        test_dataloader=get_train_test_dataloader(test_df)
        store_best_model_lstm(model,test_dataloader)
    except Exception as e:
        logger.log(logging.ERROR,f"An error occurred in main: {e}")
        raise

if __name__=="__main__":
    main()

