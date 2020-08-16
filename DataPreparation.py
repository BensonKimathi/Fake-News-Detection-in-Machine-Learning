# %load DataPreparation.py
#importing of libraries
import pandas as pd #Data processing, CSV file
import numpy as numpy #Linear algebra
import seaborn as sns #For making plots
import nltk
from nltk.stem import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

#Load the data
test_filename = 'test.csv'
train_filename = 'train.csv'
valid_filename = 'valid.csv'

train_data = pd.read_csv(train_filename)
# test_data = pd.read_csv(test_filename)
# valid_data = pd.read_csv(valid_filename)

"""ROOT ='/datasets/'

train_data = pd.read_csv(ROOT+'train.csv')
test_data = pd.read_csv(ROOT+'test.csv')
valid_data = pd.read_csv(ROOT+'valid.csv')

"""

#Looking at the data - the data format and types

def data_info(): #function to hold data information
    print(train_data.info())
#     print(test_data.info())
#     print(valid_data.info())

data_info()

#Taking a look at the data
def data_obs():
    print(train_data.head())
    # print(test.head())
    # print(valid_data.head())

data_obs()

# ----- EXPLORING DATA ----- #
#Checking for Class Imbalance
def create_distribution(dataFile):
    sns.set_style('whitegrid')
    return sns.countplot(x='Label', data=dataFile)

create_distribution(train_data)
#create_distribution(test_data)
#create_distribution(valid_data)

#Checking for Data Quality (Missing values)
def data_quality():
    print("Checking for data qualities...")
    train_data.isnull().sum()
    train_data.info()

    # test_data.isnull().sum()
    # test_data.info()

    # valid_data.isnull().sum()
    # valid_data.info()

data_quality()
# The training dataset does not have missiong values therfore cleaning is not required

#Checking for Text content 
#Put the label column on a list to check the first 10 texts
def text_content():
    tweet = train_data["Statement"].to_list()
    for i in range (5):
        print('Tweet Number '+str(i+1)+': '+tweet[i])
text_content()

# ----- FINDING IMPORTANT WORDS ----- #
#Stemming

eng_stemmer = SnowballStemmer('english')
stopwords = set(nltk.corpus.stopwords.words('english'))

def stem_tokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    return stemmed

#Processing the data
def process_data(data, exclude_stopword=True,stem=True):
    tokens = [w.lower() for w in data]
    tokens_stemmed = tokens
    tokens_stemmed = stem_tokens(tokens, eng_stemmer)
    tokens_stemmed = [w for w in tokens_stemmed if w not in stopwords]
    return tokens_stemmed

#N-grams are the fusion of multiple letters or multiple words. 
#They are formed in such a way that even the previous and next words are captured

#Unigrams
def create_unigram(words):
    assert type(words) == list
    return words

#Bigrams
def create_bigrams(words):
    assert type(words) == list
    skip = 0
    join_str = " "
    Len = len(words)
    if Len > 1:
        lst = []
        for i in range(Len-1):
            for k in range(1,skip+2):
                if i+k < Len:
                    lst.append(join_str.join([words[i],words[i+k]]))
    else: #Set it as unigram
        lst = create_unigram(words)
    return lst

#trigrams
def create_trigrams(words):
    assert type(words) == list
    skip = 0
    join_str = " "
    Len = len(words)
    if Len > 2:
        lst = []
        for i in range(1,skip+2):
            for k1 in range(1, skip+2):
                for k2 in range(1,skip+2):
                    if i+k1 < Len and i+k1+k2 < Len:
                        lst.append(join_str.join([words[i],words[i+k1],words[i+k2]]))
                        # lst.append(join_str.join([words[i], words[i+k1],words[i+k1+k2])])
        else:
            #set is as bigram
            lst = create_bigram(words)
    return lst