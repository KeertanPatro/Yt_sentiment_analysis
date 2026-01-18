import pandas as pd
import boto3
from Src.download_upload_scripts import get_data_s3
from typing import Literal
import os
import json

def upload_to_s3(file_type:Literal['Untrained','Trained','Errors_Training'],file_name:str,local_filepath:str):
    s3=boto3.client(
        "s3",
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name='us-east-1'
    )
    s3_path=f'{file_type}/{file_name}'
    s3.upload_file(local_filepath,'youtube-training-data',s3_path)


def update_untrained_data(df,file):
    with open('Data/trained_comment_ids.json','r') as f:
        trained_comment_ids=json.load(f)
    df = df[~df['comment_id'].isin(trained_comment_ids)]
    df.to_parquet(file)
    file_name=file.split('/')[-1]
    upload_to_s3('Untrained',file_name,file)

def remove_files(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Removed: {file_path}")


    

def update_trained_data(df,train_df):
    with open('Data/trained_comment_ids.json','r') as f:
        trained_comment_ids=json.load(f)
    df = df[df['comment_id'].isin(trained_comment_ids)]
    if train_df is None:
        train_df=pd.DataFrame()
    df=pd.concat([df,train_df]).drop_duplicates(subset=['comment_id'])
    df.to_parquet('Data/trained_data.parquet')
    upload_to_s3('Trained','trained_data.parquet','Data/trained_data.parquet')


def main():
    file_paths= get_data_s3('Untrained','youtube-training-data')
    trained_data_paths= get_data_s3('Trained','youtube-training-data')
    df=pd.DataFrame()
    for file in file_paths:
        df=pd.concat([pd.read_parquet(file),df])
    
    update_untrained_data(df,file)

    if trained_data_paths is not None:
        train_df=pd.DataFrame()
        for file in trained_data_paths:
            train_df=pd.concat([pd.read_parquet(file)],train_df)
    else:
        train_df=None

    update_trained_data(df,train_df)
    
    upload_to_s3('Errors_Training','log_errors.csv','log_errors.csv')

    remove_files('Data/')



if __name__=='__main__':
    main()




    
    

    
    
    

    


    
    
    














# Update untrained data in S3
# keep all the utrianed data in one file
# update training data in s3
