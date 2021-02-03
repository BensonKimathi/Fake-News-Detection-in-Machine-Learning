import streamlit as st 
import joblib,os
# NLP Pkgsop=9
import spacy
# nlp = spacy.load('en')
# EDA pkgs
import pandas as pd
# Wordcloud
from wordcloud import WordCloud
from PIL import Image
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')

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

# from FeatureSelection import preprocess
# # Vectorizer
# news_vectorizer = open('Modelling.joblib',"rb")
# news_cv = joblib.load(news_vectorizer)

# Load Our Models
def load_prediction_models(model_file):
    loaded_models = joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_models

def get_keys(val,my_dict):
    for key,value in my_dict.items():
        if val == value:
            return key

def main():
    """News Classifier App with Streamlit """
    st.title("News Classifer ML App")

    all_ml_models = ["Logistic Regression","Naive Bayes","Random Forest","Decision Tree"]
    model_choice = st.sidebar.selectbox("Choose ML Model",all_ml_models)

    news_text = st.text_area("Enter Text","Type Here")
    prediction_labels = {'True':1,'Fake':0}
         
    if st.button("Classify"):
        st.text("Original test ::\n{}".format(news_text))
        # trueFake['clean_title'] = DataPreparation.true_fake['title'].apply(preprocess)
        # df_text = pd.DataFrame(['news_text'])
        # vect_text = transform(pd.DataFrame(['news_text'])).apply(preprocess) 
        # df_text = pd.DataFrame(['news_text']).astype(str)
        
        # vect_text = df_text.apply(preprocess)
        vect_text = news_cv.transform([news_text]).toarray()
        
        if model_choice == 'Logistic Regression':
            predictor = joblib.load()
            prediction = predictor.predict(vect_text)
            st.write(prediction)
            # predictor = load_prediction_models("models/logistic_regression.pkl")
        #     prediction = predictor.predict(vect_text)
            # st.write(prediction)
        # elif model_choice == 'RFOREST':
        #     predictor = load_prediction_models("models/newsclassifier_RFOREST_model.pkl")
        #     prediction = predictor.predict(vect_text)
        #     # st.write(prediction)
        # elif model_choice == 'NB':
        #     predictor = load_prediction_models("models/newsclassifier_NB_model.pkl")
        #     prediction = predictor.predict(vect_text)
        #     # st.write(prediction)
        # elif model_choice == 'DECISION_TREE':
        #     predictor = load_prediction_models("models/newsclassifier_CART_model.pkl")
        #     prediction = predictor.predict(vect_text)
        #     # st.write(prediction)

        final_result = get_keys(prediction,prediction_labels)
        st.success("News Categorized as:: {}".format(final_result))

    # if choice == 'NLP':
    #     st.info("Natural Language Processing")
    #     news_text = st.text_area("Enter Text","Type Here")
    #     nlp_task = ["Tokenization","NER","Lemmatization","POS Tags"]
    #     task_choice = st.selectbox("Choose NLP Task",nlp_task)
    #     if st.button("Analyze"):
    #         st.info("Original Text {}".format(news_text))

    #         docx = nlp(news_text)
    #         if task_choice == 'Tokenization':
    #             result = [ token.text for token in docx ]

    #         elif task_choice == 'Lemmatization':
    #             result = ["'Token':{},'Lemma':{}".format(token.text,token.lemma_) for token in docx]
    #         elif task_choice == 'NER':
    #             result = [(entity.text,entity.label_)for entity in docx.ents]
    #         elif task_choice == 'POS Tags':
    #             result = ["'Token':{},'POS':{},'Dependency':{}".format(word.text,word.tag_,word.dep_) for word in docx]

    #         st.json(result)

    #         if st.button("Tabulize"):
    #             docx = nlp(news_text)
    #             c_tokens = [ token.text for token in docx ]
    #             c_lemma = [token.lemma_ for token in docx]
    #             c_pos = [word.tag_ for word in docx]

    #             new_df = pd.DataFrame(zip(c_tokens,c_lemma,c_pos),columns=['Tokens','Lemma','POS'])
    #             st.dataframe(new_df)

    #         if st.checkbox("Wordcloud"):
    #             wordcloud =  WordCloud().generate(news_text)
    #             plt.imshow(wordcloud,interpolation='bilinear')
    #             plt.axis("off")
    #             st.pyplot()
            
if __name__ == '__main__':
	main()