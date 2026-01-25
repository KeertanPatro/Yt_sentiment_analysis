import boto3
import json
import os
from typing import Literal
s3_client= boto3.client('s3',aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),region_name='us-east-1')
s3_resource= boto3.resource('s3',aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),region_name='us-east-1')

def get_all_files(bucket_name='youtube-training-data',folder_type=None):
  files=[]
  bucket = s3_resource.Bucket(bucket_name)
  if folder_type:
    for file in bucket.objects.filter(Prefix=folder_type):
        files.append(file.key)
  else:
    for obj in bucket.objects.all():
        files.append(obj.key)
  return files

def get_data_s3(folder_type:Literal['Untrained','Trained','Errors_Training'],bucket_name='youtube-training-data',local_path="Data/"):
    files=get_all_files(bucket_name=bucket_name,folder_type=folder_type)
    local_file_paths=[]
    for file in files:
        file_name=file.split('/')[-1]
        if not os.path.exists(local_path+file_name):
            s3_client.download_file(bucket_name,file,local_path+file_name)
            local_file_paths.append(local_path+file_name)
            return local_file_paths
        else:
            print("File path already exists no need")
            local_file_paths.append(local_path+file_name)
            return local_file_paths

def save_vocab_s3(path,vocab,max_len):
    s3_path='token_info.json'
    token_info={'vocab':vocab,'max_length':max_len}
    with open(path,'w') as f:
        json.dump(token_info,f)
    s3_client.upload_file(path,'youtube-training-data',s3_path)

def upload_to_s3(file_type:Literal['Untrained','Trained','Errors_Training'],file_name:str,local_filepath:str):
    s3=boto3.client(
        "s3",
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name='us-east-1'
    )
    s3_path=f'{file_type}/{file_name}'
    s3.upload_file(local_filepath,'youtube-training-data',s3_path)
