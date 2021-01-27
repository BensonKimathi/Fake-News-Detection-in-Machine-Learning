# import libraries
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load Datasets
true=pd.read_csv('datasets/True.csv')
fake=pd.read_csv('datasets/Fake.csv')
scraped=pd.read_csv('datasets/scraped_data.csv')

# Create column with labels
fake["label"] = "fake"
true["label"] = "true"
scraped["label"] = scraped["site"].apply(lambda x: "fake" if(x == "breitbart" or x == "infowars" or x == "theonion") else "true")

#Rename published column with date
scraped.rename(columns={"published": "date"}, inplace=True) 

# Drop Labels
scraped.drop(labels=["link","site"], axis=1, inplace=True)
true.drop(labels=["subject"], axis=1, inplace=True)

#True news has location and publisher, which fake doesn't have. Remove for consistency
true["text"] = true['text'].apply(lambda x: x.partition("-")[2])

# Change date column to datetime
true['date'] = pd.to_datetime(true['date'].str.replace(',', ''))
true["date"] = pd.to_datetime(true['date'], format='%Y-%m-%d', utc=False)
fake['date'] = pd.to_datetime(fake['date'].str.replace(',', ''))
fake["date"] = pd.to_datetime(fake['date'], format='%Y-%m-%d', utc=False)
scraped["date"] = pd.to_datetime(true.date, format='%Y-%m-%d', utc=False)

#Combine scraped with current datasets
scraped_t = scraped[scraped["label"] == "true"]
scraped_f = scraped[scraped["label"] == "fake"]
true = pd.concat([true,scraped_t], axis=0, ignore_index=True)
fake = pd.concat([fake,scraped_f], axis=0, ignore_index=True)

# Combine both datasets to one
true_fake = pd.concat([true, fake]).reset_index(drop = True)

# Combine Title with Text
true_fake['combined'] = true_fake['title'] + ' ' + true_fake['text']

# Set target values as 'fake'=0 and 'true'=1
true_fake["target"] = true_fake["label"].apply(lambda x: 1 if(x == "true") else 0)

# Drop rows with null values
true_fake.dropna(subset=['text', 'date'],inplace=True)

#Shuffle the dataset
true_fake.sample(frac=1)

# Save the cleaned and ready-to-use dataset to a new CSV
true_fake = pd.DataFrame(true_fake) 
# --To-Do
# Some code to save my dataset to Google Drive. Do you know how? Kindly help

# true_fake.to_csv(../\datasets\true_fake_data.csv', index=False) 