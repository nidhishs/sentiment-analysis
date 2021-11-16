from flask import Flask, request, jsonify
from model import *
from dotenv import load_dotenv
import requests
import re
import os
import demoji

load_dotenv('auth.env')

app = Flask('SentimentBird')
model = get_e2e()


def demojify(input_text):
    text_emojis = {
        ':-*\)': 'smile', ':-*]': 'smile', ':-*d': 'smile',
        ':-*\(': 'frown', ':-*\[': 'frown', ':-*/': 'unsure',
        ':-*o': 'astonish', ':-*0': 'astonish', 'xd': 'laugh',
        ';-*\)': 'wink', ":'\(": 'cry', ':3': 'smile', '&lt;3': 'love',
    }
    # Find all icon emojis
    icon_emojis = demoji.findall(input_text)
    emojis = {**text_emojis, **icon_emojis}

    for emoji, emoji_text in emojis.items():
        # Add extra space to avoid combining the text with the next word.
        # Extra space is removed later.
        input_text = re.sub(emoji, f' {emoji_text} ', input_text)

    return input_text


def standardize_text(input_text):

    # Convert to lower case
    input_text = input_text.lower()

    # Fix word lengthening
    input_text = re.sub(r"(.)\1{2,}", r"\1\1", input_text)

    # Remove all URLs, hashtags, mentions
    input_text = re.sub(r'(https|http)?:\/\/\S+', ' ', input_text)
    input_text = re.sub(r'^#\w+|\s#\w+', ' ', input_text)
    input_text = re.sub(r'^@\w+|\s@\w+', ' ', input_text)

    # Convert all emojis to their text counterparts
    input_text = demojify(input_text)

    # Convert HTML references to text
    input_text = re.sub(r'&amp;', 'and ', input_text)
    input_text = re.sub(r'&quot;', '', input_text)
    input_text = re.sub(r'&gt;', '', input_text)

    # Remove non-ASCII characters
    input_text = re.sub(r'\w*[^\x00-\x7F]+\w*', ' ', input_text)

    # Remove additional spaces
    input_text = re.sub(r'\s\s+', ' ', input_text)
    input_text = input_text.strip()

    return input_text


def query_tweets(tag):
    TOKEN = os.environ.get('bearer-token')
    headers = {"Authorization": f"Bearer {TOKEN}"}
    search_url = f"https://api.twitter.com/2/tweets/search/recent?query=%23{tag}%20lang%3Aen"

    response = requests.request("GET", search_url, headers=headers)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)

    return response.json()


def predict(text):
    predictions = model.predict(text)
    predictions = predictions.flatten()
    predictions = [0 if pred < 0.5 else 1 for pred in predictions]
    return predictions


@app.route('/predict_text/', methods=['POST'])
def predict_text():
    input_text = request.form.get('text')
    texts = [standardize_text(input_text)]
    predictions = predict(texts)
    output = [{'body': input_text, 'sentiment': predictions[0]}]

    response = jsonify(output)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/predict_tag/', methods=['POST'])
def predict_tag():
    tag = request.form.get('tag')
    query_data = query_tweets(tag)['data']
    tweets = list()
    for data in query_data:
        tweets.append(data['text'])
    standardized_tweets = [standardize_text(tweet) for tweet in tweets]

    predictions = predict(standardized_tweets)
    output = list()
    for tweet, prediction in zip(tweets, predictions):
        output.append({'body': tweet, 'sentiment': prediction})

    print(output)
    response = jsonify(output)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == '__main__':
    app.run(debug=True)