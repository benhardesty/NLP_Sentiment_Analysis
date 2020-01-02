import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

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

    text = re.sub('[^A-Za-z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stopwords_set]
    text = [lemmatizer.lemmatize(word) for word in text]
    return ' '.join(text)


print("start load data:", time.asctime(time.localtime(time.time())))

pos_reviews = pd.read_csv('test_pos.csv')
neg_reviews = pd.read_csv('test_neg.csv')
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

file = open("CountVectorizer.pickle","rb")
cv = pickle.load(file)
file.close()

print("start transform:", time.asctime(time.localtime(time.time())))
X = cv.transform(X)
print("end transform:", time.asctime(time.localtime(time.time())))

print("start test models:", time.asctime(time.localtime(time.time())))
file = open("multinomialNB.pickle","rb")
multinomialNB = pickle.load(file)
file.close()
multinomialNBPredictions = multinomialNB.predict(X)
print("MultinomialNB")
print(confusion_matrix(y,multinomialNBPredictions))
print(classification_report(y,multinomialNBPredictions))

file = open("logisticRegression.pickle","rb")
logisticRegression = pickle.load(file)
file.close()
logisticRegressionPredictions = logisticRegression.predict(X)
print("LogisticRegression")
print(confusion_matrix(y,logisticRegressionPredictions))
print(classification_report(y,logisticRegressionPredictions))

file = open("sgdClassifier.pickle","rb")
sgdClassifier = pickle.load(file)
file.close()
sgdClassifierPredictions = sgdClassifier.predict(X)
print("SGDClassifier")
print(confusion_matrix(y,sgdClassifierPredictions))
print(classification_report(y,sgdClassifierPredictions))

file = open("linearSVC.pickle","rb")
linearSVC = pickle.load(file)
file.close()
linearSVCPredictions = linearSVC.predict(X)
print("LinearSVC")
print(confusion_matrix(y,linearSVCPredictions))
print(classification_report(y,linearSVCPredictions))

file = open("randomForestClassifier.pickle","rb")
randomForestClassifier = pickle.load(file)
file.close()
randomForestClassifierPredictions = randomForestClassifier.predict(X)
print("RandomForestClassifier")
print(confusion_matrix(y,randomForestClassifierPredictions))
print(classification_report(y,randomForestClassifierPredictions))

# Overall
predictions = multinomialNBPredictions + logisticRegressionPredictions + sgdClassifierPredictions + linearSVCPredictions + randomForestClassifierPredictions
confidence = []

for i, prediction in enumerate(predictions):
    confidence.append(prediction/5)
    predictions[i] = 1 if prediction >= 3 else 0

print("Overall vote:")
print(confusion_matrix(y,predictions))
print(classification_report(y,predictions))
print(accuracy_score(y,predictions))
print("end test models:", time.asctime(time.localtime(time.time())))
