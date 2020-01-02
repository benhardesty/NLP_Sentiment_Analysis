import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report

import re
import time
import pickle

import nltk
from nltk.stem import WordNetLemmatizer

stopwords_set = set(nltk.corpus.stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

posReviews = pd.read_csv('posElectronics.csv')
negReviews = pd.read_csv('negElectronics.csv')

train_pos = posReviews[:400000]
train_neg = negReviews[:400000]

test_pos = posReviews[400000:]
test_neg = negReviews[400000:]

train_pos.to_csv('train_pos.csv')
train_neg.to_csv('train_neg.csv')
test_pos.to_csv('test_pos.csv')
test_neg.to_csv('test_neg.csv')
