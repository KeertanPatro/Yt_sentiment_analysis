import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import dataset,DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
import json


device="cuda" if torch.cuda.is_available() else "cpu"


class Lstm_model(nn.Module):
  def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim):
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
  


logger=logging.getLogger("model_building")
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

def load_data(train_path,test_path):
    try:
        train_df=pd.read_csv(train_path)
        test_df=pd.read_csv(test_path)
        return train_df, test_df
    except FileNotFoundError as e:
        logger.log(logging.ERROR, f"FileNotFoundError: {e} - The specified file was not found")
        raise
    except Exception as e:
        logger.log(logging.ERROR, f"An error occurred while loading data: {e}")
        raise




def get_tokenizer(df,type="test"):
    try:
        tokenizer=Tokenizer(num_words=2000,lower=True)
        tokenizer.fit_on_texts(df['new_clean_comment'])
        logger.log(logging.INFO,"Tokenizer created successfully")
        return tokenizer
    except Exception as e:
       logger.log(logging.ERROR,f"An error occured while getting tokenizer: {e}")
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


def get_train_test_dataloader(train_df,test_df,batch_size=4):
    try:
        X_train,y_train,vocab_size=text_to_vec(train_df,type="train")
        X_test,y_test=text_to_vec(test_df)
        X_train=torch.tensor(X_train,dtype=torch.long).to(device)
        y_train=torch.tensor(y_train,dtype=torch.long).to(device)
        X_test=torch.tensor(X_test,dtype=torch.long).to(device)
        y_test= torch.tensor(y_test,dtype=torch.long).to(device)
        train_dataset=TensorDataset(X_train,y_train)
        test_dataset=TensorDataset(X_test,y_test)
        batch_size=4
        train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
        test_dataloader=DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

        logger.log(logging.INFO,"Train and test dataloaders created successfully")
        return train_dataloader,test_dataloader,vocab_size
    except Exception as e:
        logger.log(logging.ERROR,f"An error occurred while creating train and test dataloaders: {e}")
        raise

def train_model(train_dataloader,vocab_size,embedding_dim=128,hidden_dim=64):
   try:
    output_dim=3
    model=Lstm_model(vocab_size=vocab_size,embedding_dim=embedding_dim,hidden_dim=hidden_dim,output_dim=output_dim)
    model.to(device)
    criterion=nn.CrossEntropyLoss()
    lr=0.001
    optimizer=optim.Adam(model.parameters(),lr=lr)
    num_epochs=1
    model.train()
    logger.log(logging.INFO,"Training started")
    for epoch in range(num_epochs):
            total_loss=0
            for x,y in tqdm(train_dataloader):
                y_pred=model(x)
                y_pred=y_pred.squeeze()
                batch_size=y.shape[0]
                y_pred=y_pred.view(batch_size,output_dim)
                loss=criterion(y_pred,y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss+=loss.item()
            print(f"Loss for epoch-{epoch} is", total_loss)
    logger.log(logging.INFO,f"mTrsining complete Final loss- is {total_loss}")
   except Exception as e:
    logger.log(logging.ERROR,f"An error occurred while training the model: {e}")
    raise
        
   return model

def save_model(model_config,model):
   try:
      model_path=model_config['model_path']
      torch.save(model.state_dict(), model_path)
      logger.log(logging.INFO,f"Model saved at {model_path}")
   except:
      logger.log(logging.ERROR,"Error in saving the Model")
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



def get_root_directory():
   try:
      path=os.path.dirname(os.path.abspath(__file__))
      prev_path=os.path.abspath(os.path.join(path,"../../"))
      return prev_path
   except Exception as e:
      logger.log(logging.ERROR,f"Error in getting root directory: {e}")
      raise

def main():
   try:
      dir_path=get_root_directory()
      train_data_path=dir_path+"/Data/Clean/train_clean.csv"
      test_data_path=dir_path+"/Data/Clean/test_clean.csv"
      model_path=dir_path+"/lstm_model.pth"
      train_df,test_df=load_data(train_data_path,test_data_path)
      logger.log(logging.INFO,"Data loaded successfully")
      train_dataloader,test_dataloader,vocab_size=get_train_test_dataloader(train_df,test_df)
      model=train_model(train_dataloader,vocab_size)
      model_config={"model_path":model_path,"vocab_size":vocab_size,"embedding_dim":128,"hidden_dim":64}
      json.dump(model_config,open(dir_path+"/model_config.json","w"))
      save_model(model_config,model)
   except Exception as e:
      logger.log(logging.ERROR,f"Error in building the model, {e}")
      raise

if __name__=="__main__":
   main()