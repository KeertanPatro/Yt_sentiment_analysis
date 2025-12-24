from fastapi import FastAPI,Request
from Src.Data_scripts.data_preprocess import pre_process,remove_stopwords
from Src.Model_scripts.model_building import Lstm_model,get_tokenizer
import json
import pandas as pd
import logging
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import dataset,DataLoader,TensorDataset
import io
import base64
import torch
import os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
load_dotenv()




YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY')
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'


category_mapping={0:'Postive',1:'Neutral',2:'Negative'}
df=pd.read_csv('Data/Clean/train_clean.csv')
app=FastAPI()
configs=json.load(open('model_config.json','r'))
vocab_size=configs['vocab_size']
embedding_dim=configs['embedding_dim']
hidden_dim=configs['hidden_dim']
model=Lstm_model(vocab_size,embedding_dim=embedding_dim,hidden_dim=hidden_dim,output_dim=3)
model.load_state_dict(torch.load('./lstm_model.pth'))
tokenizer=get_tokenizer(df)



logger=logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('log_errors.csv')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


def get_youtube_comments(video_id,max_results=100):
    try:
        youtube = build(
            YOUTUBE_API_SERVICE_NAME, 
            YOUTUBE_API_VERSION,
            developerKey=YOUTUBE_API_KEY
        )
        comments = []
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
                    'published_at': comment['publishedAt']
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
        return comments

    except HttpError as e:
        print(f"YouTube API Error: {e}")
        raise Exception(f"Failed to fetch comments: {str(e)}")
    






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

    


def text_to_vec(text,max_len=900):
    try:
        try:
            X=tokenizer.texts_to_sequences(text)
        except:
            logger.log(logging.ERROR,"Error in converting text to sequence")
            raise
        try:
            X=pad_sequences(X,maxlen=max_len)
            X=torch.tensor(X,dtype=torch.long)
        except Exception as e:
            logger.log(logging.ERROR,f"Error in padding sequences:{e}")
            raise
        return X

    except Exception as e:
        logger.log(logging.ERROR,f"An error occurred while converting text to vector: {e}")
        raise

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
        
    fig, ax = plt.subplots(figsize=(8, 6))

    labels = list(sentiment_counts.keys())
    sizes = list(sentiment_counts.values())
    colors = ['#10b981', '#ef4444', '#6b7280']  # green, red, gray
    explode = (0.1, 0.1, 0)
    
    ax.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='%1.1f%%', shadow=True, startangle=90)
    ax.axis('equal')
    ax.set_title('Sentiment Distribution', fontsize=16, fontweight='bold')
    
    # Save to base64
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    return f"data:image/png;base64,{image_base64}"





YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY')
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'


@app.post("/analyze")
async def predict(request:Request):
    print("/here!!!!")
    data=await request.json()
    video_id=data['video_id']
    comments=get_youtube_comments(video_id)
    text=[]
    for comment in comments:
        text.append(comment['text'])
    wordcloud_base64=generate_wordcloud(text)
    sentiment_chart_base64=generate_sentiment(text)
    response = {
            'success': True,
            'video_id': video_id,
            'total_comments': len(comments),
            'sentiment_chart': sentiment_chart_base64,
            'wordcloud': wordcloud_base64
        }
    return response