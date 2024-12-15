# 1439 were taken from the processed data and manually labeled
# 2500 was taken from https://portals.mdi.georgetown.edu/public/stance-detection-KE-MLM

# Remaining training was developed from the following script on data from https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet
import torch

# Using CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

# Initializing VADER and SentiWordNet models
vader_analyzer = SentimentIntensityAnalyzer()

# Sentiment helper functions
# VADER model
def vader_sentiment(text):
    return vader_analyzer.polarity_scores(text)['compound']

# TextBlob model
def textblob_sentiment(text):
    return TextBlob(text).sentiment.polarity

# SentiWordNet model
def sentiwordnet_sentiment(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    sentiment_score = 0

    for token in tokens:
        word = lemmatizer.lemmatize(token, pos=get_wordnet_pos(token))
        synsets = list(swn.senti_synsets(word))
        if synsets:
            sentiment = synsets[0]  # Use the first synset
            sentiment_score += sentiment.pos_score() - sentiment.neg_score()

    return 1 if sentiment_score > 0 else -1

def get_wordnet_pos(word):
    """Map POS tag to first character for lemmatization."""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {'J': wordnet.ADJ, 'N': wordnet.NOUN, 'V': wordnet.VERB, 'R': wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# Combined sentiment analysis
def combined_sentiment_analysis(text):
    scores = []

    # Aggregate sentiment from all models
    scores.append(1 if vader_sentiment(text) > 0 else (0 if vader_sentiment(text) == 0 else -1))  # VADER
    scores.append(1 if textblob_sentiment(text) > 0 else (0 if textblob_sentiment(text) == 0 else -1))  # TextBlob
    scores.append(1 if sentiwordnet_sentiment(text) > 0 else (0 if sentiwordnet_sentiment(text) == 0 else -1))  # SentiWordNet

    # Count votes
    positive_votes = scores.count(1)
    negative_votes = scores.count(-1)
    neutral_votes = scores.count(0)

    # Determine majority sentiment
    if positive_votes > negative_votes and positive_votes > neutral_votes:
        return 1  # Positive sentiment
    elif negative_votes > positive_votes and negative_votes > neutral_votes:
        return -1  # Negative sentiment
    else:
        return 0  # Neutral sentiment

# Text preprocessing
def preprocess_tweets(tweets_df):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    def clean_text(text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#', '', text)
        text = re.sub(r'[^A-Za-z0-9\s]', '', text)
        words = word_tokenize(text)
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return " ".join(words)

    tweets_df['cleaned_tweet'] = tweets_df['tweet'].apply(clean_text)
    return tweets_df

# Load datasets
trump_csv_filepath = "./training_data/hashtag_donaldtrump.csv"
biden_csv_filepath = "./training_data/hashtag_joebiden.csv"
tweets_biden = pd.read_csv(biden_csv_filepath, lineterminator='\n', parse_dates=True)
tweets_trump = pd.read_csv(trump_csv_filepath, lineterminator='\n', parse_dates=True)

# Mapping sentiment analysis to pro-Biden (0), pro-Trump (1), neutral/other (2)
def map_biden_sentiment(score):
    """Map sentiment for Biden tweets: + -> 0, - -> 1, Neutral -> 2."""
    if score == 1:
        return 0  # Positive
    elif score == -1:
        return 1  # Negative
    elif score == 0:
        return 2  # Neutral

def map_trump_sentiment(score):
    """Map sentiment for Trump tweets: + -> 1, - -> 0, Neutral -> 2."""
    if score == 1:
        return 1  # Positive
    elif score == -1:
        return 0  # Negative
    elif score == 0:
        return 2  # Neutral

# Preprocessing tweets
tweets_biden = preprocess_tweets(tweets_biden)
tweets_trump = preprocess_tweets(tweets_trump)
print('Preprocessing DONE')

# Applying combined sentiment analysis
tweets_biden['raw_sentiment'] = tweets_biden['cleaned_tweet'].apply(combined_sentiment_analysis)
tweets_biden.to_csv('biden_sent_raw.csv', index=False)
print('Biden Sentiment DONE')
tweets_trump['raw_sentiment'] = tweets_trump['cleaned_tweet'].apply(combined_sentiment_analysis)
tweets_trump.to_csv('trump_sent_raw.csv', index=False)
print('Trump Sentiment DONE')

# Map sentiment labels for each dataset
tweets_biden['sentiment'] = tweets_biden['raw_sentiment'].apply(map_biden_sentiment)
tweets_biden.to_csv('biden_sent.csv', index=False)
tweets_trump['sentiment'] = tweets_trump['raw_sentiment'].apply(map_trump_sentiment)
tweets_trump.to_csv('trump_sent.csv', index=False)
print('Mapping DONE')

# Combining dataframes
combined_df = pd.concat([tweets_biden, tweets_trump])
combined_df.rename(columns={'tweet_id': 'id', 'cleaned_tweet': 'text', 'sentiment': 'poli_label'}, inplace=True)
combined_df = combined_df[['id', 'text', 'poli_label']]

manual_df = pd.read_csv('./training_data/manual_train.csv') # Manually labeled data from 2024
final_df = pd.concat([combined_df, manual_df], ignore_index=True)
final_df.dropna(inplace=True)

final_df.to_csv('./training_data/training.csv', index=False)