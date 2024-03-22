import numpy as np
import pandas as pd
import re
#import emoji
#from textblob import TextBlob
#import nltk
#from nltk.tokenize import word_tokenize,sent_tokenize
#nltk.download('punkt')
#from nltk.corpus import stopwords
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('vader_lexicon')
#from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
#from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
#from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score,f1_score
import streamlit as st
import pickle
result = None

st.title("Flipkart based Review Detection using ML")
text = st.text_input("Enter the text")

import os
# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Construct the path to the pickle file relative to the current script
pickle_file_path = os.path.join(current_dir, "sentiment.pkl")

# Load the pickle file
with open(pickle_file_path, "rb") as f:
    model = pickle.load(f)

if st.button("Submit")==True:
    result = model.predict([text])[0]
    st.write(result)

if result == 'Positive':
    st.image("https://reviewpoint.com/wp-content/uploads/2023/08/review-1024x536-1.jpg")
elif result == 'Negative':
    st.image("https://www.callcentrehelper.com/images/stories/2018/02/negative-feedback-thumbs-760.png")