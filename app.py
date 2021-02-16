# Imports
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re
import string
from nltk.tokenize import word_tokenize
from PIL import Image

# Constants
# Logistic regression is the best performing model, so we use it for prediction
MODEL_PATH = 'models/'
MODEL_FILE_NAME = 'logistic_regression.sav'
RANDOM_STATE = 42

# Load pipeline
@st.cache(allow_output_mutation=True)
def load_pipeline(model_path=MODEL_PATH, model_file_name=MODEL_FILE_NAME):
    """
    Load the Text Processing and Classifier Pipeline
    """
    return pickle.load(open(model_path + model_file_name, 'rb'))

pipeline = load_pipeline()

def preprocess(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)  
    return text

st.title('Machine Learning News Classifier')

image = Image.open('fake_news.jpeg')
st.image(image, caption='Can this be Fake or True News?',use_column_width=True, output_format='auto')

st.write("""
        Enter the text of a news story and a trained logistic regression
        classifier will classify it as Truthful or Fake. Please note that the
        algorithm is not checking the facts of the news story, it is basing
        the classification on the style of the text of the story; specifically, it
        is basing the classification only on the stop words (common words) in
        the story.
         """)

news_story = st.text_area('Enter the text of a News Article', height=400)

if st.button('Classify'):
    tokens = preprocess(news_story)
    class_ = pipeline.predict([tokens])

    if class_ == 0:
        class_text = 'Fake'
    else:
        class_text = 'True'

    probability = round(pipeline.predict_proba([tokens])[0][class_][0] * 100, 2)
    st.subheader('Classification Results')
    st.write('This story could be', class_text, 'with a probability score of',probability, '% .')
