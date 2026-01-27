import torch
import os
import logging
import json
import boto3


s3_client= boto3.client('s3',aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),region_name='us-east-1')

def get_token_s3(path,bucket_name):
    try:
      s3_client.download_file(bucket_name,path,'Data/vocab.json')
      with open('Data/vocab.json','r') as f:
        token_info=json.load(f)
        vocab=token_info['vocab']  
      return vocab
    except Exception as e:
      raise 

vocab=get_token_s3(path='token_info.json',bucket_name='youtube-training-data')
max_length=97
max_vocab_length=46578



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
                if len(word)<30:
                  if word not in self.vocab:
                      if len(self.vocab)>=max_vocab_length:
                          status=True
                          break
                      self.vocab[word]=len(self.vocab)
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


tokenizer=Tokenizer(vocab=vocab,max_len=max_length)

def text_to_vec(text):
    try:
        X=tokenizer.texts_to_sequences(text)
        X=torch.tensor(X,dtype=torch.long)
        return X
    except:
        raise