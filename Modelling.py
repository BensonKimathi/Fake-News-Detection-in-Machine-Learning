import FeatureSelection
import DataPreparation

import pandas as pd
import numpy as np
import seaborn as sns
import joblib
import pickle
import matplotlib.pyplot as plt

import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression

# Convert Text to Vectors
x = DataPreparation.df["text"]
y = DataPreparation.df["target"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# LOGISTIC REGRESSION
pipe = Pipeline([('count_vectorization', CountVectorizer()),
                 ('tfidf_vectorization', TfidfTransformer()),
                 ('LR', LogisticRegression())])

LR = pipe.fit(x_train, y_train)
pred_lr = LR.predict(x_test)

LR.score(x_test, y_test)
print(classification_report(y_test, pred_dt))
#saving this model to the disk
model_file = 'models/logistic_regression.sav'
pickle.dump(LR,open(model_file,'wb'))

# DECISION TREE
pipe = Pipeline([('count_vectorization', CountVectorizer()),
                 ('tfidf_vectorization', TfidfTransformer()),
                 ('DT', DecisionTreeClassifier())])

DT = pipe.fit(x_train, y_train)
pred_dt = DT.predict(x_test)

DT.score(x_test, y_test)
print(classification_report(y_test, pred_dt))
#saving this model to the disk
model_file = 'models/decison_tree.sav'
pickle.dump(DT,open(model_file,'wb'))

# RANDOM FOREST
pipe = Pipeline([('count_vectorization', CountVectorizer()),
                 ('tfidf_vectorization', TfidfTransformer()),
                 ('RFC', RandomForestClassifier())])

RFC = pipe.fit(x_train, y_train)
pred_rfc = RFC.predict(x_test)

RFC.score(x_test, y_test)
print(classification_report(y_test, pred_rfc))
#saving this model to the disk
model_file = 'models/random_forest.sav'
pickle.dump(RFC,open(model_file,'wb'))
