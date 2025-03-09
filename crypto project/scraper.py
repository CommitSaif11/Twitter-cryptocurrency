import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
import schedule
import time

# Load positive and negative words
with open('positive-words.txt', 'r') as file:
    positive_words = set(file.read().splitlines())

with open('negative-words.txt', 'r') as file:
    negative_words = set(file.read().splitlines())

# Preprocess the tweets
def preprocess_tweet(tweet):
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    tweet = re.sub(r'[^A-Za-z0-9 ]+', '', tweet)
    return tweet.lower()

# Sentiment analysis
def sentiment_score(tweet):
    words = tweet.split()
    pos_score = sum(1 for word in words if word in positive_words)
    neg_score = sum(1 for word in words if word in negative_words)
    return pos_score - neg_score

# Load the data and train the model
data = pd.read_csv('XLM-tweets-processed.csv')
data = data.dropna()
data = data.reset_index(drop=True)
data['cleaned_tweet'] = data.iloc[:, 1].apply(preprocess_tweet)
data['sentiment_score'] = data['cleaned_tweet'].apply(sentiment_score)
data['positive'] = data['cleaned_tweet'].apply(lambda x: sum(1 for word in x.split() if word in positive_words))
data['negative'] = data['cleaned_tweet'].apply(lambda x: sum(1 for word in x.split() if word in negative_words))
X = data[['positive', 'negative', 'sentiment_score']]
y = data.iloc[:, 2]
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Trading strategy
def trading_strategy(tweet):
    processed_tweet = preprocess_tweet(tweet)
    sentiment = sentiment_score(processed_tweet)
    features = pd.DataFrame([[sum(1 for word in processed_tweet.split() if word in positive_words),
                              sum(1 for word in processed_tweet.split() if word in negative_words),
                              sentiment]], columns=['positive', 'negative', 'sentiment_score'])
    prediction = model.predict(features)
    return 'Buy' if prediction == 1 else 'Sell'

# Scrape tweets from a Twitter account
def scrape_tweets(username):
    url = f'https://twitter.com/{username}'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    tweets = [p.text for p in soup.find_all('p', class_='TweetTextSize')]
    return tweets

# Automate the process
def automate_trading():
    username = 'crypto_account'  # Replace with the actual Twitter account username
    tweets = scrape_tweets(username)
    for tweet in tweets:
        action = trading_strategy(tweet)
        print(f'Tweet: {tweet}\nTrading Action: {action}\n')

# Schedule the script to run every hour
schedule.every().hour.do(automate_trading)

while True:
    schedule.run_pending()
    time.sleep(1)
