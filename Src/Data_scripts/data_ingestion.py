import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import yaml
import os

logger=logging.getLogger("data_ingestion")
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





def load_params(params_path:str)->dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        return params
    except FileNotFoundError as e:
        logger.log(logging.ERROR, f"FileNotFoundError: {e} - The specified parameters file was not found")
        raise
    except yaml.YAMLError as e:
        logger.log(logging.ERROR, f"YAMLError: {e} - Error parsing the YAML file")
        raise
    except Exception as e:
        logger.log(logging.ERROR, f"An error occurred while loading parameters: {e}")
        raise


def clean_data(df):
    try:
        df.dropna(inplace=True)
        df=df.drop_duplicates()
        df=df[df['clean_comment'].str.strip()!='']
        return df
    except KeyError as e:
        logger.log(logging.ERROR, f"KeyError: {e} - Column not found in DataFrame")
        raise
    except Exception as e:
        logger.log(logging.ERROR, f"An error occurred while cleaning data: {e}")
        
        raise

def split_data(df,test_size):
    try:
        train, test = train_test_split(df, test_size=test_size, random_state=42)
        return train, test
    except Exception as e:
        logger.log(logging.ERROR, f"An error occurred while splitting data: {e}")
        # log the exception
        raise


def save_data(train,test,data_path):
    try:
        data_path=os.path.join(data_path, "Raw")
        os.makedirs(data_path, exist_ok=True)
        train.to_csv("Data/Raw/train.csv", index=False)
        test.to_csv("Data/Raw/test.csv", index=False)
        # log that data has been saved
    except Exception as e:
        logger.log(logging.ERROR, f"An error occurred while saving data: {e}")
        # log the exception
        raise

def main():
    try:
        params= load_params("params.yaml")
        try:
            test_size=params.get('data_ingestion').get('test_size', 0.2)
        except Exception as e:
            logger.log(logging.ERROR, f"An error occurred while accessing test_size from params: {e}")
        try:
            df=pd.read_csv("https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv")
        except Exception as e:
            logger.log(logging.ERROR, f"An error occurred while loading the dataset: {e}")
            raise
        df=clean_data(df)
        train, test=split_data(df, test_size=test_size)
        save_data(train, test,"Data/")
    except Exception as e:
        logger.log(logging.ERROR, f"An error occurred in main: {e}")
        # log the exception
        print(f"An error occurred in main: {e}")

if __name__ == "__main__":
    main()


