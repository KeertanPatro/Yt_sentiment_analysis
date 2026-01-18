import mlflow
from mlflow.tracking import MlflowClient
import torch
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from torch.utils.data import dataset,DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
import json




mlflow.set_tracking_uri("http://ec2-52-23-156-179.compute-1.amazonaws.com:5000")
client = MlflowClient()
models = client.search_registered_models()
source_path=models[0].latest_versions[0].source
current_model = mlflow.pytorch.load_model(
    model_uri=source_path,
    map_location="cpu"  
)


class Tokenizer:
    def __init__(self,vocab,max_len):
        self.vocab=vocab
        self.max_len=max_len
        self.pad=vocab['<pad>']
        self.unk=vocab['<unk>']
    
    def fit_on_texts(self,texts:list):
        status=False
        for text in texts:
            for word in text.split():
                word=word.strip()
                if word not in self.vocab:
                    self.vocab[word]=len(self.vocab)
                    if len(self.vocab)>=46578:
                        break
            if status:
                break
            
                
    
        

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


def text_to_vec(df,tokenizer):
    try:
        try:            
            X=tokenizer.texts_to_sequences(df['clean_comment'])
        except:
            raise
        y=np.array(df['category'])
        return X,y
    except Exception as e:
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


df=pd.read_csv('https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')
df.dropna(inplace=True)
df['category']=df['category'].map({0:0,1:1,-1:2})
vocab=json.load(open('/Users/keertan.patro/Desktop/Practice/TrainingPipelines/Data/token_info.json','r'))
vocab=vocab['vocab']
max_len=97
tokenizer=Tokenizer(vocab,max_len)

X=np.array(tokenizer.texts_to_sequences(df['clean_comment'].to_list()))
y=np.array(df['category'].to_list())


device='cpu'
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,shuffle=True,stratify=y)
X_train=torch.tensor(X_train,dtype=torch.long).to(device)
X_test=torch.tensor(X_test,dtype=torch.long).to(device)
y_train=torch.tensor(y_train,dtype=torch.long).to(device)
y_test=torch.tensor(y_test,dtype=torch.long).to(device)
train_dataset=TensorDataset(X_train,y_train)
test_dataset=TensorDataset(X_test,y_test)
batch_size=64
output_dim=3
train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_dataloader=DataLoader(test_dataset,batch_size=batch_size,shuffle=True)
print("Here!!!")
report_old,y_true_old,y_pred_old=evaluate_model(current_model,test_dataloader)
print(report_old)