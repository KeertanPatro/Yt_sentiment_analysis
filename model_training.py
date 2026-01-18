import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import dataset,DataLoader,TensorDataset
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import json
import yaml
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
from Src.download_upload_scripts import get_data_s3,get_token_s3,save_vocab_s3
import yaml
try:
  load_dotenv()
except:
  pass

with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)
num_epochs=params.get('model_training').get('num_epochs')
batch_size=params.get('model_training').get('num_epochs')

model_path="cardiffnlp/twitter-roberta-base-sentiment-latest"

tokenizer_pretrained = AutoTokenizer.from_pretrained(model_path)

model_Master = AutoModelForSequenceClassification.from_pretrained(model_path)




mlflow.set_tracking_uri("http://ec2-52-23-156-179.compute-1.amazonaws.com:5000")
client = MlflowClient()
models = client.search_registered_models()
source_path=models[0].latest_versions[0].source
current_model = mlflow.pytorch.load_model(
    model_uri=source_path,
    map_location="cpu"  
)
print("Here")




nltk.download('stopwords')
nltk.download('wordnet')
lemm=WordNetLemmatizer()

stopwords=[word for word in stopwords.words('english') if word not in ('not','but','however','no','yet')]





logger=logging.getLogger("model_training")
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

device="cuda" if torch.cuda.is_available() else "cpu"



  



vocab,max_length=get_token_s3(path='token_info.json',bucket_name='youtube-training-data')
if not vocab:
   vocab={'<pad>':0,'<unk>':1}
   max_length=None
max_length=97
    




def remove_stopwords(text):
  new_text=""
  for word in text.split():
    if word not in stopwords:
      word=lemm.lemmatize(word)
      new_text+=word+" "
  return new_text.strip()

def pre_process(text):
  url_pattern="https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F\][0-9a-fA-F]))+"
  non_ascii_pattern="[^A-Za-z0-9\s!?.,]"
  url_matches=re.findall(url_pattern,text)
  non_ascii_matches=re.findall(non_ascii_pattern,text)
  text=text.lower()
  text=text.replace("\n"," ")
  text=text.strip()
  if url_matches:
    for url in url_matches:
      text=text.replace(url,"")
  if non_ascii_matches:
    for non_ascii in non_ascii_matches:
      text=text.replace(non_ascii,"")
  text=text.strip()
  text=remove_stopwords(text)
  if len(text.split())<=2:
    return np.nan
  return text


class Tokenizer:
    def __init__(self,vocab,max_len):
        self.vocab=vocab
        self.max_len=max_len
        self.pad=vocab['<pad>']
        self.unk=vocab['<unk>']
    
    def fit_on_texts(self,texts:list):
        for text in texts:
            for word in text.split():
                word=word.strip()
                if word not in self.vocab:
                    self.vocab[word]=len(self.vocab)
    
        

    def texts_to_sequences(self,texts:list):
        sequences=[]
        for text in texts:
            seq=[]
            for word in text.split():
                word=word.strip()
                if len(seq)==self.max_len:
                    break
                elif word in self.vocab:
                    seq.append(self.vocab[word])
                else:
                    seq.append(self.unk)
            seq=seq+[self.pad]*(self.max_len-len(seq))
            sequences.append(seq)
        return sequences





def text_to_vec(df,tokenizer,type="test"):
    try:
        try:
            if type=="train":
              tokenizer.fit_on_texts(df['comment_text'].to_list())               
            X=tokenizer.texts_to_sequences(df['comment_text'])
        except:
            logger.log(logging.ERROR,"Error in converting text to sequence")
            raise
        y=np.array(df['sentiment_category'])
        return X,y
    except Exception as e:
        logger.log(logging.ERROR,f"An error occurred while converting text to vector: {e}")
        raise



    




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







def get_train_test_dataloader(train_df,test_df,batch_size,tokenizer):
    try:
        
        X_train,y_train=text_to_vec(df=train_df,type="train",tokenizer=tokenizer)
        X_test,y_test=text_to_vec(df=test_df,tokenizer=tokenizer)
        X_train=torch.tensor(X_train,dtype=torch.long).to(device)
        y_train=torch.tensor(y_train,dtype=torch.long).to(device)
        X_test=torch.tensor(X_test,dtype=torch.long).to(device)
        y_test= torch.tensor(y_test,dtype=torch.long).to(device)
        train_dataset=TensorDataset(X_train,y_train)
        test_dataset=TensorDataset(X_test,y_test)
        train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
        test_dataloader=DataLoader(test_dataset,batch_size=batch_size,shuffle=True)
        logger.log(logging.INFO,"Train and test dataloaders created successfully")
        return train_dataloader,test_dataloader
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
    num_epochs=3
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


def upload_model_to_mlflow(model):
  try:
    with mlflow.start_run():
      mlflow.pytorch.log_model(
          model,
          artifact_path="model",
          registered_model_name="sentiment_pytorch_model"
      )
    client.transition_model_version_stage(
      name="sentiment_pytorch_model",
      stage="Production",
      version=1,
      archive_existing_versions=True
    )
    logger.log(logging.INFO,"Model uploaded to MLflow successfully")
    return True
  except Exception as e:
    logger.log(logging.ERROR,f"An error occurred while uploading model to MLflow, {e}")
    raise

def get_data_labelled(df):
   # get non labelled Data
   global model_Master
   global tokenizer_pretrained
   temp_df=df.loc[df['sentiment_category'].isna()]
   if temp_df.empty:
      logger.log(logging.INFO,"No unlabelled data found")
      df['sentiment_category']=df['sentiment_category'].map({0:0,1:1,-1:2})
      del model_Master
      del tokenizer_pretrained
      return df
   else:
        with torch.no_grad():
            pairs=df['comment_text'].to_list()
            inputs = tokenizer_pretrained(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = model_Master(**inputs, return_dict=True).logits.view(-1, ).float()
            scores=scores.reshape(len(pairs),-1)
            sent_cat=torch.argmax(scores,dim=1)
            df['sentiment_category']=sent_cat.numpy()
            df['sentiment_category']=temp_df['sentiment_category'].map({0:2,1:1,2:0})
            del model_Master
            del tokenizer_pretrained
            del temp_df
            return df
   
   



def main():
  local_file_paths=get_data_s3(folder_type="Untrained")
  try:
    df=pd.DataFrame()
    for file in local_file_paths:
      temp_df=pd.read_parquet(file)
      df=pd.concat([df,temp_df])
    logger.log(logging.INFO,"Data read successfully from parquet files")
  except Exception  as e:
    logger.log(logging.ERROR,f"An error occurred while reading data from parquet files: {e}")
    raise
  
  # label data
  df=get_data_labelled(df)

  try:
    train_df,test_df=train_test_split(df,test_size=0.2,random_state=42)
    train_df['comment_text']=train_df['comment_text'].apply(pre_process)
    test_df['comment_text']=test_df['comment_text'].apply(pre_process)
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)
    logger.log(logging.INFO,"Data split and preprocessed successfully")
  except Exception as e:
    logger.log(logging.ERROR,f"An error occurred while splitting and preprocessing data: {e}")
    raise

  try:
    comment_ids_trained=train_df['comment_id'].to_list()
    with open('Data/trained_comment_ids.json','w') as f:  
      json.dump(comment_ids_trained,f)
    logger.log(logging.INFO,"Trained comment ids saved successfully")
  except Exception as e:
    logger.log(logging.ERROR,f"An error occurred while saving trained comment ids: {e}")
    raise
  if not max_length:
     comments_len=[len(comment) for comment in train_df['comment_text']]
     max_len=int(np.percentile(comments_len,95))
  else:
     max_len=max_length
  tokenizer=Tokenizer(vocab,max_len)
  train_dataloader,test_dataloader=get_train_test_dataloader(train_df,test_df,batch_size,tokenizer)
  
  try:
    save_vocab_s3(path='Data/token_info.json',vocab=tokenizer.vocab,max_len=tokenizer.max_len)
    logger.log(logging.INFO,"Vocab saved to S3 successfully")
  except Exception as e:
    logger.log(logging.ERROR,f"An error occurred while saving vocab to S3: {e}")
    raise
  
  try:
    new_model=train_model(train_dataloader,len(tokenizer.vocab))
  except Exception as e:
    logger.log(logging.ERROR,f"An error occurred while training the model: {e}")
    raise
  del train_dataloader
  del tokenizer
  report_old,y_true_old,y_pred_old=evaluate_model(current_model,test_dataloader)
  report_new,y_true_new,y_pred_new=evaluate_model(new_model,test_dataloader)
  
  
  if report_old['accuracy']<report_new['accuracy']:
    mlflow.pytorch.log_model(new_model,artifact_path="lstm_model")
    logger.log(logging.INFO,"New model logged to MLflow as it outperformed the existing model")
  else:
     print("Existing model outperformed the new model. No update made.")
     logger.log(logging.INFO,"Existing model outperformed the new model. No update made.")


if __name__=="__main__":
    main()
    





    








    















    