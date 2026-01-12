import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model_path="cardiffnlp/twitter-roberta-base-sentiment-latest"
print("Here")
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("There")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
print("Now")
model.eval()

pairs = ['Covid cases are increasing fast!',"Apple is such a good company to work at","Leverage EDu plays lot of politics and does let the employes grow!"]
df=pd.read_csv('https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')

print("hetre")
temp_df=df[:23]
with torch.no_grad():
    pairs=temp_df['clean_comment'].to_list()
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    scores=scores.reshape(len(pairs),-1)
    sent_cat=torch.argmax(scores,dim=1)
    temp_df['sentiment_category']=sent_cat.numpy()
    temp_df['sentiment_category']=temp_df['sentiment_category'].map({0:2,1:1,2:0})
    temp_df.to_csv('test_sentiment_output.csv',index=False)
    

print(model.config.id2label)