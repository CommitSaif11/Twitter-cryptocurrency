import tkinter as tk
from tkinter import messagebox
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import schedule
import time
import threading
import pathlib
from PIL import Image, ImageTk

# Load positive and negative words
positive_words_path = pathlib.Path(__file__).parent / 'positive-words.txt'
negative_words_path = pathlib.Path(__file__).parent / 'negative-words.txt'

with open(positive_words_path, 'r') as file:
    positive_words = set(file.read().splitlines())

with open(negative_words_path, 'r') as file:
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
data_path = pathlib.Path(__file__).parent / 'XLM-tweets-processed.csv'
data = pd.read_csv(data_path)
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

# Evaluate the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

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
    tweets = [p.text for p in soup.find_all('div', {'data-testid': 'tweetText'})]
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

# GUI Functionality
def on_predict():
    tweet = tweet_entry.get()
    if tweet:
        action = trading_strategy(tweet)
        messagebox.showinfo("Trading Action", f'Trading Action: {action}')
    else:
        messagebox.showwarning("Input Error", "Please enter a tweet.")

# Create the GUI
root = tk.Tk()
root.title("Crypto Trading Strategy")

# Load the background image
bg_image_path = pathlib.Path(__file__).parent / 'pngtree-crypto-twitter-header-bitcoin-photos---picture-image_15689764.jpg'
bg_image = Image.open(bg_image_path)
bg_image = bg_image.resize((800, 600), Image.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)

# Create a canvas to place the background image
canvas = tk.Canvas(root, width=800, height=600)
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, image=bg_photo, anchor="nw")

# Create a frame to hold the widgets
frame = tk.Frame(root, bg='#2c3e50')
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)

title_font = ('Helvetica', 16, 'bold')
label_font = ('Helvetica', 12)
button_font = ('Helvetica', 12, 'bold')

tk.Label(frame, text="Enter Tweet:", font=label_font, bg='#2c3e50', fg='#ecf0f1').pack(pady=10)
tweet_entry = tk.Entry(frame, width=50, font=label_font)
tweet_entry.pack(pady=10)

predict_button = tk.Button(frame, text="Predict Trading Action", font=button_font, bg='#3498db', fg='#ecf0f1', command=on_predict)
predict_button.pack(pady=20)

# Run the GUI and the scheduler
def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)

scheduler_thread = threading.Thread(target=run_scheduler)
scheduler_thread.daemon = True
scheduler_thread.start()

root.mainloop()
