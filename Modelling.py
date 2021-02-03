import FeatureSelection

import joblib,os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import confusion_matrix

true_fake = FeatureSelection.trueFake

# Using Title for detection
# X_train, X_test, y_train, y_test = train_test_split(true_fake.clean_joined_title, true_fake.target, test_size = 0.2,random_state=2)
# vec_train = CountVectorizer().fit(X_train)
# X_vec_train = vec_train.transform(X_train)
# X_vec_test = vec_train.transform(X_test)

# # Logistic Regression
# model = LogisticRegression(C=2)
# model.fit(X_vec_train, y_train)
# predicted_value = model.predict(X_vec_test)
# accuracy_value = roc_auc_score(y_test, predicted_value)
# print(accuracy_value)

# USING TEXT FOR DETECTION
X_train, X_test, y_train, y_test = train_test_split(true_fake.clean_joined_text, true_fake.target, test_size = 0.2,random_state=2)
vec_train = CountVectorizer().fit(X_train)
X_vec_train = vec_train.transform(X_train)
X_vec_test = vec_train.transform(X_test)

# Logistic Regression
model = LogisticRegression(C=2.5)
model.fit(X_vec_train, y_train)
# predicted_value = model.predict(X_vec_test)
# accuracy_value = roc_auc_score(y_test, predicted_value)
# print(accuracy_value)
joblib.dump(model, 'logistic_regression.sav')

# # USING COMBINED TEXT AND TITLE
# X_train, X_test, y_train, y_test = train_test_split(true_fake.clean_joined_final, true_fake.target, test_size = 0.2,random_state=0)
# vec_train = CountVectorizer().fit(X_train)
# X_vec_train = vec_train.transform(X_train)
# X_vec_test = vec_train.transform(X_test)

# # Logistic Regression
# model = LogisticRegression(C=3)
# model.fit(X_vec_train, y_train)
# predicted_value = model.predict(X_vec_test)
# accuracy_value = roc_auc_score(y_test, predicted_value)
# print(accuracy_value)

# joblib.dump(model,'Modelling.pkl')