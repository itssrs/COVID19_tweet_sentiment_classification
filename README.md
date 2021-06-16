# COVID19_tweet_sentiment_classification

`Problem Statement`

Text Classification on the COVID19 tweet data. The tweets have been pulled from Twitter and manual tagging has been done then. The names and usernames have been given codes to avoid any privacy concerns

https://www.kaggle.com/datatattle/covid-19-nlp-text-classification

`Step to train model`

> * Download the Repo
> * Run src/training.py file to generate train and test data save to Dataset directory.
> * Run command to start the training `python -m spacy train config/config.cfg --output ./output`

`Step to test model using Streamlit`

> * Run the main.py file `streamlit run main.py`
> * Below images are the result 

![image](https://user-images.githubusercontent.com/62031889/122270976-872ebe00-cefc-11eb-9125-b78515863255.png)

![image](https://user-images.githubusercontent.com/62031889/122271013-931a8000-cefc-11eb-8fa6-111e8a24fca6.png)

![image](https://user-images.githubusercontent.com/62031889/122271044-9a418e00-cefc-11eb-94f7-fc52899da564.png)
