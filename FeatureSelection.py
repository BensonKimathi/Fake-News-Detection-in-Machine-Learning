# Import Libraries
import DataPreparation
import pandas as pd

import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load Dataset
true_fake = pd.read_csv('datasets/true_fake_data.csv')

# Drop null in text column
true_fake.dropna(subset=['text'],inplace=True)

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

true_fake["text"] = true_fake["text"].apply(preprocess)