import DataPreparation

import nltk
import re
from nltk.corpus import stopwords
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2 and token not in stop_words:
            result.append(token)
            
    return result

# Using Title for Prediction
trueFake = DataPreparation.true_fake
trueFake['clean_title'] = DataPreparation.true_fake['title'].apply(preprocess)
trueFake['clean_joined_title']=trueFake['clean_title'].apply(lambda x:" ".join(x))

# Using Text for Prediction
trueFake['clean_text'] = DataPreparation.true_fake['text'].apply(preprocess)
trueFake['clean_joined_text']=trueFake['clean_text'].apply(lambda x:" ".join(x))

# Using Joined Title and Text
trueFake['clean_final'] = DataPreparation.true_fake['combined'].apply(preprocess)
trueFake['clean_joined_final']=trueFake['clean_final'].apply(lambda x:" ".join(x))