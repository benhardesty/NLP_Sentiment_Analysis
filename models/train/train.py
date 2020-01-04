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

def clean_review(text):
    """
    Clean a review so it can be used for ML.
    1. Remove punctuation.
    2. Put all letters to lowercase.
    3. Split the string into an array of words.
    4. Lemmatize each word.
    5. Join the words back together and return the string.
    """

    nopunc = re.sub('[^A-Za-z]', ' ', text) # Remove all characters other than letters.
    nopunc = nopunc.lower()
    words = nopunc.split()
    words = [word for word in words if word not in stopwords_set]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

print("start load data:", time.asctime(time.localtime(time.time())))

pos_reviews = pd.read_csv('train_pos.csv')
neg_reviews = pd.read_csv('train_neg.csv')
reviews = pd.concat([pos_reviews,neg_reviews]).sample(frac=1)
print("end load data:", time.asctime(time.localtime(time.time())))


print("start clean data:", time.asctime(time.localtime(time.time())))
reviews.dropna(subset=['reviewText'], inplace=True)
# ReviewTextClean column already added by separateReviewsBySentiment.py file
# reviews['reviewTextClean'] = reviews['reviewText'].apply(clean_review)
reviews.dropna(subset=['reviewTextClean'],inplace=True)
print("end clean data:", time.asctime(time.localtime(time.time())))

X = reviews['reviewTextClean']
y = reviews['sentiment']

print("start fit:", time.asctime(time.localtime(time.time())))
cv = CountVectorizer(max_features=20000)
cv.fit(X)
print("end fit:", time.asctime(time.localtime(time.time())))

# Save the CountVectorizer.
file = open("CountVectorizer.pickle","wb")
pickle.dump(cv, file)
file.close()

print("start transform:", time.asctime(time.localtime(time.time())))
# Transform the reviewTextClean column to a vector count.
X = cv.transform(X)
print("end transform:", time.asctime(time.localtime(time.time())))

print("start training models:", time.asctime(time.localtime(time.time())))
# MultinomialNB
multinomialNB = MultinomialNB()
multinomialNB.fit(X,y)
file = open("multinomialNB.pickle","wb")
pickle.dump(multinomialNB, file)
file.close()

# LogisticRegression
logisticRegression = LogisticRegression()
logisticRegression.fit(X,y)
file = open("logisticRegression.pickle","wb")
pickle.dump(logisticRegression, file)
file.close()

# SGDClassifier
sgdClassifier = SGDClassifier()
sgdClassifier.fit(X,y)
file = open("sgdClassifier.pickle","wb")
pickle.dump(sgdClassifier, file)
file.close()

# LinearSVC
linearSVC = LinearSVC()
linearSVC.fit(X,y)
file = open("linearSVC.pickle","wb")
pickle.dump(linearSVC, file)
file.close()

# RandomForestClassifier
randomForestClassifier = RandomForestClassifier()
randomForestClassifier.fit(X,y)
file = open("randomForestClassifier.pickle","wb")
pickle.dump(randomForestClassifier, file)
file.close()
print("end training models:", time.asctime(time.localtime(time.time())))
