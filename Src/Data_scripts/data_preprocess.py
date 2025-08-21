import pandas as pd
import numpy as np
import logging
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


lemm=WordNetLemmatizer()

def pre_process(text):
  url_pattern="https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F\][0-9a-fA-F]))+"
  non_ascii_pattern="[^\x00-\x7F]+"
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