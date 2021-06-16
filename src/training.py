import os
os.chdir('E:\\Projects\\Coronavirus_tweets_NLP-Text_Classification')
import numpy as np
import pandas as pd
import spacy
import re
import string
from spacy.tokens import DocBin
import pre_processing

nlp = spacy.load("en_core_web_sm")
num_texts = 41157 #size of train dataset

if __name__ == "__main__":

    # Load raw data
    train_raw_data = pd.read_csv("Dataset/Corona_NLP_train.csv")
    test_raw_data = pd.read_csv("Dataset/Corona_NLP_test.csv")

    # dropping inetger feature
    train_data = train_raw_data[['OriginalTweet','Sentiment']].dropna()
    test_data = test_raw_data[['OriginalTweet','Sentiment']].dropna()

    # Converting Sentiment
    sentiment = {"Extremely Positive":"Positive", "Extremely Negative":"Negative",
            "Positive":"Positive","Negative":"Negative","Neutral":"Neutral"}

    train_data["Sentiment"] = train_data.Sentiment.map(sentiment)
    test_data["Sentiment"] = test_data.Sentiment.map(sentiment)

    # Pre-process training and test data
    train_data.OriginalTweet = train_data.OriginalTweet.apply(pre_processing.remove_emoji)
    train_data.OriginalTweet = train_data.OriginalTweet.apply(pre_processing.remove_url)
    train_data.OriginalTweet = train_data.OriginalTweet.apply(pre_processing.clean_text)
    test_data.OriginalTweet = test_data.OriginalTweet.apply(pre_processing.remove_emoji)
    test_data.OriginalTweet = test_data.OriginalTweet.apply(pre_processing.remove_url)
    test_data.OriginalTweet = test_data.OriginalTweet.apply(pre_processing.clean_text)

    # converting training data into spacy format data
    train_data['tuples'] = train_data.apply(lambda row: (row['OriginalTweet'], row['Sentiment']), axis=1)
    train_spacy = train_data.tuples.tolist()

    test_data['tuples'] = test_data.apply(lambda row: (row['OriginalTweet'], row['Sentiment']), axis=1)
    test_spacy = test_data.tuples.tolist()

    # saving train and test data to disk
    train_docs = pre_processing.make_docs(train_spacy[:num_texts])
    doc_bin = DocBin(docs=train_docs)
    doc_bin.to_disk("E:\\Projects\\Coronavirus_tweets_NLP-Text_Classification\\Dataset\\train.spacy")

    test_docs = pre_processing.make_docs(test_spacy[:num_texts])
    doc_bin = DocBin(docs=test_docs)
    doc_bin.to_disk("E:\\Projects\\Coronavirus_tweets_NLP-Text_Classification\\Dataset\\test.spacy")