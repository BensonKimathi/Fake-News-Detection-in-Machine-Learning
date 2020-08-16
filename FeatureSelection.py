"""

"""

import DataPreparation
import pandas as import pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from gensim.models.word2vec import Word2Vec

#Create the transform
vectorizer = CountVectorizer()
#Tokenizing #Encoding the transformed document
train_vector = vectorizer.fit_transform(DataPreparation.train_data['Statement'])

print (vectorizer)
print (train_vector)

#Document term matrix
def get_Vectorizer_stats():
    #Vocabulary size
    train_vector.shape

    #Check the vocabulary
    print(vectorizer.vocabulary_)

#TF-IDF Features
tfidf = TfidTransformer()
train_tfidf = tfidf.fit_transform(train_vector)

def get_tfidf_stats():
     #Vocabulary size
    train_tfidf.shape

    #Check the vocabulary
    print(train_tfidf.A[:10])

#Bag of words - ngrams
countV_ngram = CountVectorizer(ngram_range=(1,3),stop_words='english')
tfidf_ngram = TfidTransformer(use_idf=True,smooth_idf=True)

tfidf_ngram = TfidfVectorizer(stop_words='english',ngram_range=(1,4),use_idf=True,smooth_idf=True)

#Part Of Speech (POS) Tagging
tagged_sentences = nltk.corpus.treebank.tagged_sents()

cutoff = int(.75 *len(tagged_sentences))
training_sentences = DataPreparation.train_data['Statement']

print(training_sentences)

"""
stop_words = set(stopwords.words('english'))
# Tokenize the text
tokens = sent_tokenize(DataPreparation.train_data['Statement'])
#Generate tagging for all the tokens using loop
for i in tokens:    
    words = nltk.word_tokenize(i)    
    words = [w for w in words if not w in stop_words]    
#  POS-tagger.    
tags = nltk.pos_tag(words)
tags
"""
#Training POS tagger based on words
def features(sentence, index):
    """ sentence: [w1,w2, ...], index: the index of the word """
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': [index == len(sentence)-1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'capitals_inside': sentence[index][1:].lower()!=sentence[index][1:],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index-1],
        'next_word': '' if index == len(sentence)-1 else sentence[index+1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdgit()
    }