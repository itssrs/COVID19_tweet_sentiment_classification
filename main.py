import streamlit as st
import spacy
from src import pre_processing

def preprocess(text):
    text = pre_processing.remove_emoji(text)
    text = pre_processing.remove_url(text)
    text = pre_processing.clean_text(text)
    return text

def predict(text):
    model = spacy.load("output/model-last")
    pred = model(text)
    return pred

if __name__ == "__main__":

    st.title("COVID19 Tweet Sentiment Classifier")
    tweet_input = st.text_area("Enter Text","Type Here")

    if tweet_input != '':

        # Pre-process tweet
        sentence = preprocess(tweet_input)

        # Prediction
        with st.spinner('Predicting...'):
            prediction = predict(sentence)

        for key, value in prediction.cats.items():
            if value >= 0.5:
                res = key
                percentage = value
                break

        st.write('Prediction:')
        st.write(res + ' with {:.2f}'.format(percentage*100), '% confidence.')