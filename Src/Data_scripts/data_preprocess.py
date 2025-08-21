import pandas as pd
import numpy as np
import logging
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stopwords=[word for word in stopwords.words('english') if word not in ('not','but','however','no','yet')]

logger=logging.getLogger("data_preprocess")
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

lemm=WordNetLemmatizer()

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
  if len(text.split())<=2:
    return np.nan
  return text

def remove_stopwords(text):
  new_text=""
  for word in text.split():
    if word not in stopwords:
      word=lemm.lemmatize(word)
      new_text+=word+" "
  return new_text.strip()

def load_data(data_path):
  try:
    train_df=pd.read_csv(data_path+"/train.csv")
    test_df=pd.read_csv(data_path+"/test.csv")
    return train_df,test_df
  except FileNotFoundError as e:
    logger.log(logging.ERROR, f"FileNotFoundError: {e} - The specified data file was not found")
    raise
  except Exception as e:
    logger.log(logging.ERROR, f"An error occurred while loading data: {e}")
    raise

def clean_data(train_df,test_df):
    try:
        train_df['new_clean_comment']=train_df['clean_comment'].apply(pre_process)
        test_df['new_clean_comment']=test_df['clean_comment'].apply(pre_process)
        train_df.dropna(inplace=True)
        test_df.dropna(inplace=True)
        train_df['processed_text']=train_df['new_clean_comment'].apply(remove_stopwords)
        test_df['processed_text']=test_df['new_clean_comment'].apply(remove_stopwords)
        return train_df, test_df
    except KeyError as e:
        logger.log(logging.ERROR, f"KeyError: {e} - Column not found in DataFrame")
        raise
    except Exception as e:
        logger.log(logging.ERROR, f"An error occurred while cleaning data: {e}")
        raise


def save_data(train_df,test_df,output_path):
   try:
      data_path=os.path.join(output_path,"Clean")
      os.makedirs(data_path,exist_ok=True)
      train_df.to_csv(os.path.join(data_path,"train_clean.csv"),index=False)
      test_df.to_csv(os.path.join(data_path,"test_clean.csv"),index=False)
   except FileNotFoundError as e:
        logger.log(logging.ERROR, f"FileNotFoundError: {e} - The specified output directory was not found")
        raise
   except Exception as e:
        logger.log(logging.ERROR, f"An error occurred while saving data: {e}")
        raise


def main():
   try:
      data_path="Data/Raw"
      train_df,test_df=load_data(data_path)
      train_df,test_df=clean_data(train_df,test_df)
      save_data(train_df,test_df,output_path="Data/")
   except Exception as e:
        logger.log(logging.ERROR, f"An error occurred in the main function: {e}")
        raise

if __name__ =="__main__":
   main()
