import pandas as pd
import gzip
import json

import re
import nltk
from nltk.stem import WordNetLemmatizer

stopwords_set = set(nltk.corpus.stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

posReviews = []
neutReviews = []
negReviews = []

rpath = 'Electronics_5.json.gz'
wpathPos = 'posElectronics.csv'
wpathNeut = 'neutralElectronics.csv'
wpathNeg = 'negElectronics.csv'

negCount = 0
neutCount = 0
posCount = 0

def clean_review(text):
    text = re.sub('[^A-Za-z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stopwords_set]
    text = [lemmatizer.lemmatize(word) for word in text]
    return ' '.join(text)

with gzip.open(rpath, 'rb') as file:
    for i, line in enumerate(file):
        review = json.loads(line)
        if review['overall'] > 3 and posCount <= negCount:
            if 'reviewText' in review:
                review['reviewTextClean'] = clean_review(review['reviewText'])
                posReviews.append(review)
                posCount += 1
        elif review['overall'] == 3:
            if 'reviewText' in review:
                review['reviewTextClean'] = clean_review(review['reviewText'])
                neutReviews.append(review)
                neutCount += 1
        elif review['overall'] < 3:
            if 'reviewText' in review:
                review['reviewTextClean'] = clean_review(review['reviewText'])
                negReviews.append(review)
                negCount += 1


        if i % 100000 == 0:
            print("row:",i)
            print("counts | positive:", posCount, ", neutral:", neutCount, ", negative:", negCount)

posReviews = pd.DataFrame(posReviews)
neutReviews = pd.DataFrame(neutReviews)
negReviews = pd.DataFrame(negReviews)

posReviews.dropna(subset=['reviewText'],inplace=True)
neutReviews.dropna(subset=['reviewText'],inplace=True)
negReviews.dropna(subset=['reviewText'],inplace=True)

posReviews['sentiment'] = posReviews['overall'].apply(lambda x: 1 if x > 3 else 0)
neutReviews['sentiment'] = neutReviews['overall'].apply(lambda x: 1 if x > 3 else 0)
negReviews['sentiment'] = negReviews['overall'].apply(lambda x: 1 if x > 3 else 0)

posReviews.to_csv(wpathPos)
neutReviews.to_csv(wpathNeut)
negReviews.to_csv(wpathNeg)

print("Pos:")
print(posReviews['overall'].value_counts())
print(posReviews['sentiment'].value_counts())
print(posReviews.head())

print("Neut:")
print(neutReviews['overall'].value_counts())
print(neutReviews['sentiment'].value_counts())
print(neutReviews.head())

print("Neg:")
print(negReviews['overall'].value_counts())
print(negReviews['sentiment'].value_counts())
print(negReviews.head())
