# Miscelaneous
import sys
import re
import emoji
from string import punctuation
from unidecode import unidecode
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Mac Bug
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
#nltk.download('stopwords')

# Keras
from keras.optimizers import Adam, SGD
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.layers import (Dense,
                          Flatten,
                          LSTM,
                          GRU,
                          Dropout,
                          Input)
from keras.models import Sequential, Model, load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras import backend as B

# SKLearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.utils import shuffle

# Gensim
from gensim.models import KeyedVectors

class TweetSA:
    path = '../data/'
    path_word2vec = '../word2vec/'
    path_models = '../models/'
    word2vec_google = path_word2vec + 'GoogleNews-vectors-negative300.bin'
    emoji_path = path_word2vec + 'Emoji_Sentiment_Data_v1.0.csv'
    train_3_path = path + 'data3.csv'
    train_7_path = path + 'data7.csv'
    smileys = {
      ":‑)": "smiley",
      ":-]": "smiley",
      ":-3": "smiley",
      ":->": "smiley",
      "8-)": "smiley",
      ":-}": "smiley",
      ":)": "smiley",
      ":]": "smiley",
      ":3": "smiley",
      ":>": "smiley",
      "8)": "smiley",
      ":}": "smiley",
      ":o)": "smiley",
      ":c)": "smiley",
      ":^)": "smiley",
      "=]": "smiley",
      "=)": "smiley",
      ":-))": "smiley",
      ":‑D": "smiley",
      "8‑D": "smiley",
      "x‑D": "smiley",
      "X‑D": "smiley",
      ":D": "smiley",
      "8D": "smiley",
      "xD": "smiley",
      "XD": "smiley",
      ">:D": "smiley",
      ":‑(": "sad",
      ":‑c": "sad",
      ":‑<": "sad",
      ":‑[": "sad",
      ":(": "sad",
      ":c": "sad",
      ":<": "sad",
      ":[": "sad",
      ":-||": "sad",
      ">:[": "sad",
      ":{": "sad",
      ":@": "sad",
      ">:(": "sad",
      ":'‑(": "sad",
      ":'(": "sad",
      ":‑P": "playful",
      "X‑P": "playful",
      "x‑p": "playful",
      ":‑p": "playful",
      ":‑Þ": "playful",
      ":‑þ": "playful",
      ":‑b": "playful",
      ":P": "playful",
      "XP": "playful",
      "xp": "playful",
      ":p": "playful",
      ":Þ": "playful",
      ":þ": "playful",
      ":b": "playful",
      "<3": "love"
    }
    contractions = {
      "ain't": "is not",
      "amn't": "am not",
      "aren't": "are not",
      "can't": "cannot",
      "'cause": "because",
      "couldn't": "could not",
      "couldn't've": "could not have",
      "could've": "could have",
      "daren't": "dare not",
      "daresn't": "dare not",
      "dasn't": "dare not",
      "didn't": "did not",
      "doesn't": "does not",
      "don't": "do not",
      "e'er": "ever",
      "em": "them",
      "everyone's": "everyone is",
      "finna": "fixing to",
      "gimme": "give me",
      "gonna": "going to",
      "gon't": "go not",
      "gotta": "got to",
      "hadn't": "had not",
      "hasn't": "has not",
      "haven't": "have not",
      "he'd": "he would",
      "he'll": "he will",
      "he's": "he is",
      "he've": "he have",
      "how'd": "how would",
      "how'll": "how will",
      "how're": "how are",
      "how's": "how is",
      "I'd": "I would",
      "I'll": "I will",
      "I'm": "I am",
      "i'm": "I am",
      "I'm'a": "I am about to",
      "I'm'o": "I am going to",
      "isn't": "is not",
      "it'd": "it would",
      "it'll": "it will",
      "it's": "it is",
      "I've": "I have",
      "kinda": "kind of",
      "let's": "let us",
      "mayn't": "may not",
      "may've": "may have",
      "mightn't": "might not",
      "might've": "might have",
      "mustn't": "must not",
      "mustn't've": "must not have",
      "must've": "must have",
      "needn't": "need not",
      "ne'er": "never",
      "o'": "of",
      "o'er": "over",
      "ol'": "old",
      "oughtn't": "ought not",
      "shalln't": "shall not",
      "shan't": "shall not",
      "she'd": "she would",
      "she'll": "she will",
      "she's": "she is",
      "shouldn't": "should not",
      "shouldn't've": "should not have",
      "should've": "should have",
      "somebody's": "somebody is",
      "someone's": "someone is",
      "something's": "something is",
      "that'd": "that would",
      "that'll": "that will",
      "that're": "that are",
      "that's": "that is",
      "there'd": "there would",
      "there'll": "there will",
      "there're": "there are",
      "there's": "there is",
      "these're": "these are",
      "they'd": "they would",
      "they'll": "they will",
      "they're": "they are",
      "they've": "they have",
      "this's": "this is",
      "those're": "those are",
      "'tis": "it is",
      "'twas": "it was",
      "wanna": "want to",
      "wasn't": "was not",
      "we'd": "we would",
      "we'd've": "we would have",
      "we'll": "we will",
      "we're": "we are",
      "weren't": "were not",
      "we've": "we have",
      "what'd": "what did",
      "what'll": "what will",
      "what're": "what are",
      "what's": "what is",
      "what've": "what have",
      "when's": "when is",
      "where'd": "where did",
      "where're": "where are",
      "where's": "where is",
      "where've": "where have",
      "which's": "which is",
      "who'd": "who would",
      "who'd've": "who would have",
      "who'll": "who will",
      "who're": "who are",
      "who's": "who is",
      "who've": "who have",
      "why'd": "why did",
      "why're": "why are",
      "why's": "why is",
      "won't": "will not",
      "wouldn't": "would not",
      "would've": "would have",
      "y'all": "you all",
      "you'd": "you would",
      "you'll": "you will",
      "you're": "you are",
      "you've": "you have",
      "Whatcha": "What are you",
      "luv": "love",
      "sux": "sucks"
    }
    characters = ['\'', '"', '_', '-', '.', ';', ':', '?', '!', '#']
    stopwords = set(stopwords.words('english')) - set(('not', 'no'))
    labels = {
        -3: 'Very negative emotional state',
        -2: 'Moderately negative emotional state',
        -1: 'Slightly negative emotional state',
         0: 'Neural or mixed emotional state', 
         1: 'Slightly positive emotional state',
         2: 'Moderately positive emotional state',
         3: 'Very positive emotional state'
    }
    
    def __init__(self):  
        train_3_data = pd.read_csv(self.train_3_path, sep='\t', names=['text', 'class'])
        train_7_data = pd.read_csv(self.train_7_path, sep='\t', names=['text', 'class'])
        emoji_df = pd.read_csv(self.emoji_path)
        self.emoji_dict = {}
        for index, row in emoji_df.iterrows():
            self.emoji_dict[row['Emoji']] = row['Unicode name'].lower()
        train_3_data['text'] = train_3_data['text'].apply(self.standardization)
        train_7_data['text'] = train_7_data['text'].apply(self.standardization)
        all_tweets = train_3_data['text'].append(train_7_data['text'])
        self.MAX_SEQUENCE_LENGTH = int(0.7 * all_tweets.apply(lambda x: len(x)).max())
        self.tokenizer = Tokenizer(filters=' ')
        self.tokenizer.fit_on_texts(all_tweets)
        self.word_index = self.tokenizer.word_index
        dependencies = {'distance': self.distance}
        self.model = load_model("{}{}".format(self.path_models, "best_model2.h5"), custom_objects=dependencies)
        
    def distance(self, y_true, y_pred):
        return B.square(B.argmax(y_true) - B.argmax(y_pred))  
        
    def standardization(self, tweet):
        tknzr = TweetTokenizer()
        tweet = tknzr.tokenize(tweet)
        tweet = [self.smileys[i] if i in self.smileys else i for i in tweet]
        tweet = [self.emoji_dict[i] if i in self.emoji_dict else i for i in tweet]
        tweet = [self.contractions[i] if i in self.contractions else i for i in tweet]
        tweet = ' '.join(tweet)
        tweet = re.sub(r'(\\u[0-9A-Fa-f]+)', ' ', tweet)
        tweet = re.sub(r'[^\x00-\x7F]+', ' ', tweet)
        tweet = re.sub(r'http\S+', '', tweet)
        tweet = re.sub(r"@\w+", '', tweet)
        for ch in self.characters:
            tweet = tweet.replace(ch, ' ')
        tweet = tweet.lower()
        tweet = tknzr.tokenize(tweet)
        tweet = [i for i in tweet if (i not in self.stopwords) and (i not in punctuation)]
        tweet = ' '.join(tweet)
        return tweet
    
    def predict(self, tweet):
        tweet = self.standardization(tweet)
        tweet = self.tokenizer.texts_to_sequences([tweet])
        tweet = pad_sequences(tweet, maxlen=self.MAX_SEQUENCE_LENGTH)
        y = self.model.predict(tweet)[0].argmax() - 3
        return y, self.labels[y]