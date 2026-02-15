from fastapi import FastAPI,Request
import json
import pandas as pd
import logging
import io
import base64
import torch
import os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
try:
    load_dotenv()
except:
    pass
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import mlflow
from mlflow.tracking import MlflowClient
from Src.model_building import text_to_vec
from Src.download_upload_s3 import get_data_s3,upload_to_s3
executor=ThreadPoolExecutor(max_workers=4)



mlflow.set_tracking_uri("http://ec2-52-23-156-179.compute-1.amazonaws.com:5000")
client = MlflowClient()
models = client.search_registered_models()
source_path=models[0].latest_versions[0].source
model = mlflow.pytorch.load_model(
    model_uri=source_path,
    map_location="cpu"  
)


YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY')
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'


category_mapping={0:'Positive',1:'Neutral',2:'Negative'}
app=FastAPI()



logger=logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('log_errors.csv')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


def collect_and_upload_data(s3_comments):
    df=pd.DataFrame()
    untrained_data_paths=get_data_s3(folder_type='Untrained')
    for path in untrained_data_paths:
        temp_df=pd.read_parquet(path)
        df=pd.concat([temp_df,df])
        new_df=pd.DataFrame(s3_comments)
        df=pd.concat([new_df,df])
        df=df.loc[~df.duplicated(subset=['comment_id'],keep='first')]
        df.loc[:, 'created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df.loc[:, 'updated_at'] = pd.to_datetime(df['updated_at'], errors='coerce')
        df.to_parquet('Data/Untrained_comments.parquet')
        upload_to_s3('Untrained',file_name='Untrained_comments.parquet',local_filepath='Data/Untrained_comments.parquet')
    print("Upload done!!")
    os.remove('Data/Untrained_comments.parquet')

def sentiment_score_calc(val):
    y_pred=model(val)
    sentiment_dict=Counter(torch.max(y_pred,-1).indices.view(y_pred.shape[1]).tolist())
    sentiment_score=0
    for sentiment in sentiment_dict:
        if sentiment==2:
            sentiment_score+=sentiment_dict[sentiment]*-1
        elif sentiment==0:
            sentiment_score+=sentiment_dict[sentiment]*0.5
        else:
            sentiment_score+=sentiment_dict[sentiment]*1
    return sentiment_score

def get_sentiment_score_time(new_comments):
    date_month_gr={}
    for comment in new_comments:
        if comment['date_month'] not in date_month_gr:
            date_month_gr[comment['date_month']]=[comment['text']]
        else:
            date_month_gr[comment['date_month']].append(comment['text'])

    for date_month in date_month_gr:
        val=text_to_vec(date_month_gr[date_month])
        sentiment_score=sentiment_score_calc(val)
        date_month_gr[date_month]=sentiment_score
    return date_month_gr

def get_youtube_comments(video_id,max_results=100):
    try:
        youtube = build(
            YOUTUBE_API_SERVICE_NAME, 
            YOUTUBE_API_VERSION,
            developerKey=YOUTUBE_API_KEY
        )
        
        comments = []
        s3_comments=[]
        stat_request=youtube.videos().list(
            part="statistics",
            id=video_id
        )
        video_dict={}
        if stat_request:
            stat_response=stat_request.execute()
            video_dict['viewCount']=stat_response['items'][0]['statistics']['viewCount']
            video_dict['likeCount']=stat_response['items'][0]['statistics']['likeCount']
            video_dict['commentCount']=stat_response['items'][0]['statistics']['commentCount']
        
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=max_results,
            textFormat='plainText',
            order='relevance'
        )
        while request and len(comments)<100:
            response=request.execute()
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'text': comment['textDisplay'],
                    'author': comment['authorDisplayName'],
                    'likes': comment['likeCount'],
                    'published_at': comment['publishedAt'],
                    'date_month':datetime.strptime(comment['publishedAt'],"%Y-%m-%dT%H:%M:%SZ").strftime('%Y-%m')
                })
                s3_comments.append({
                    'comment_id':item['id'],
                    'video_id':video_id,
                    'comment_text':comment['textDisplay'],
                    'updated_at':comment['updatedAt'],
                    'created_at':comment['publishedAt'],
                    'training_status':0,
                    'sentiment_category':pd.NA
                })
            
            if 'nextPageToken' in response and len(comments) < max_results:
                request = youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    maxResults=max_results - len(comments),
                    pageToken=response['nextPageToken'],
                    textFormat='plainText',
                    order='relevance'
                )
            else:
                break
        executor.submit(collect_and_upload_data,s3_comments)
        del s3_comments
        video_dict['comments']=comments
        return video_dict

    except HttpError as e:
        print(f"YouTube API Error: {e}")
        raise Exception(f"Failed to fetch comments: {str(e)}")
    

def plot_sentiment_dist(sentiment_dist:dict):
    x=sentiment_dist.keys()
    y=sentiment_dist.values()
    buf = io.BytesIO()
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.xlabel("Time")
    plt.ylabel("Sentiment Score")
    plt.title("Sentiment Over Time")
    plt.savefig(buf,format="png",bbox_inches="tight")
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"




def generate_wordcloud(comments:list):
    
    text=" ".join(comments)
    if not text.strip():
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No text available', 
                ha='center', va='center', fontsize=20)
        ax.axis('off')
    else:
        wordcloud = WordCloud(
            width=800, 
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(text)
        wc_img=wordcloud.to_image()

    buffer = io.BytesIO()
    
    wc_img.save(buffer,format='png')
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{image_base64}"

    




# device='cpu'
# def get_train_test_dataloader(tests,batch_size=4):
#     try:
#         text_vec=text_to_vec(tests,type="train")
#         train_dataset=TensorDataset(X_tra)
#         test_dataset=TensorDataset(X_test,y_test)
#         batch_size=4
#         train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
#         test_dataloader=DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

#         logger.log(logging.INFO,"Train and test dataloaders created successfully")
#         return train_dataloader,test_dataloader,vocab_size
#     except Exception as e:
#         logger.log(logging.ERROR,f"An error occurred while creating train and test dataloaders: {e}")
#         raise

def generate_sentiment(texts:list):
    text_vec=text_to_vec(texts)
    y_pred=model(text_vec)
    y_pred=torch.max(y_pred,-1).indices.view(y_pred.shape[1]).tolist()
    sentiment_counts={}
    count_dict=Counter(y_pred)
    for key in count_dict:
        sentiment_counts[category_mapping[key]]=count_dict[key]
    return sentiment_counts





YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY')
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'


@app.post("/analyze")
async def predict(request:Request):
    print("/here!!!!")
    data=await request.json()
    video_id=data['video_id']
    video_dict=get_youtube_comments(video_id)
    comments=video_dict['comments']
    avg_length=round(sum([len(comment['text'].split()) for comment in comments])/len(comments),2)
    sentiment_score_dist=get_sentiment_score_time(comments)
    avg_sentiment_score=round(sum(sentiment_score_dist.values())/len(comments),2)

    text=[]
    for comment in comments:
        text.append(comment['text'])
    wordcloud_base64=generate_wordcloud(text)
    sentiment_score_dist_plot=plot_sentiment_dist(sentiment_score_dist)
    sentiment_vals=generate_sentiment(text)
    response = {
            'success': True,
            'video_id': video_id,
            'totalComments':video_dict['commentCount'],
            'totalLikes':video_dict['likeCount'],
            'avgSentiment':avg_sentiment_score,
            'avgLength':avg_length,
            'wordcloud': wordcloud_base64,
            'sentimentTime':sentiment_score_dist_plot,
            "sentimentDist":{'Positive':sentiment_vals.get('Positive',0),'Neutral':sentiment_vals.get('Neutral',0),'Negative':sentiment_vals.get('Negative',0)}
        }
    return response

@app.get("/")
def health():
    return {"message":"API is healthy and running!!"}