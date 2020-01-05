from flask import Flask, request, jsonify, render_template, send_file, session, redirect, url_for, Response
from functools import wraps
import os
import jinja2
import sys
import io
import base64
import gc
import csv

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import random
import time

import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from models.MultiModelClassifier import MultiModelClassifier
from wordcloud import WordCloud, STOPWORDS

app = Flask(__name__)
app.secret_key = "super secret key"
app.config['MAX_CONTENT_LENGTH'] = 10 * 2**20
datasets = ['Fashion', 'Software', 'Electronics']

def require_login():
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            if not session.get('logged_in'):
                # return redirect(url_for('login'))
                return f(*args, **kwargs)
            else:
                return f(*args, **kwargs)
        return wrapped
    return decorator

@app.route('/')
@require_login()
def home():
    """
    UI for home page.
    """
    return render_template('choose-analysis.html',datasets=datasets)

@app.route('/login/')
def login():
    """
    UI for login.
    """
    return render_template('login.html')

@app.route('/login/',methods=['POST'])
def authenticate():
    """
    Authenticate login.
    """
    # username = request.form['username']
    # password = request.form['password']

    # No login needed for this demo project. This is for reference for future projects only.
    loginsuccessful = True

    if loginsuccessful:
        session['logged_in'] = True
        return redirect(url_for('home'))
    else:
        output = '<label class="form-check-label" for="exampleCheck1" style="color:red;">Login was unsuccessful</label>'
        return render_template('login.html', output=output)

@app.route('/logout/')
@require_login()
def logout():
    """
    End a user's session and log out.
    """
    session['logged_in'] = False
    return redirect(url_for('home'))

@app.route('/data-exploration/<dataset>')
def analyze_dataset(dataset):
    """
    Analyze a dataset and return plots, accuracy reports, coefficient tables, and an updated dataset.
    """
    images = []
    reviews = pd.read_csv('data/{}.csv'.format(dataset.lower()),index_col=0)

    # Clean the reviews and get a frequency distribution of all the words.
    all_words = []
    reviews.dropna(subset=['reviewText'],inplace=True)
    reviews['reviewTextClean'] = reviews['reviewText'].apply(clean_review, all_words=all_words)
    reviews.dropna(subset=['reviewTextClean'],inplace=True)
    freqdist = nltk.FreqDist(all_words)

    # Create a sentiment and text length column
    reviews['actualSentiment'] = reviews['overall'].apply(lambda x: 1 if x >= 3 else 0)
    reviews['length'] = reviews['reviewText'].apply(len)

    # Predict the sentiment in the reviews.
    X = reviews['reviewTextClean']
    mmc = MultiModelClassifier(1)
    predictions, confidence = mmc.predict(X)

    # Add sentiment and confidence to the output data.
    reviews['predictedSentiment'] = predictions
    reviews['predictedSentimentConfidence'] = confidence

    datasetTable = reviews.head(10).to_html(classes='table table-striped table-hover table-sm table-responsive')

    # Create accuracy reports
    conf_matrix = confusion_matrix(reviews['actualSentiment'],predictions)
    conf_matrix = createTableFromConfusionMatrix(conf_matrix)
    class_report = classification_report(reviews['actualSentiment'],predictions)
    class_report = createTableFromClassificationReport(class_report)
    accuracy = "{}%".format(round(accuracy_score(reviews['actualSentiment'],predictions)*100,2))

    # Create tables explaining the coefficients of the linear models used by the MultiModelClassifier
    coefficientTables = createCoefficientsTables(mmc,freqdist)

    # Update actual and predicted sentiment columns so they are easier to read on the charts that will be output.
    reviews['actualSentiment'] = reviews['actualSentiment'].apply(lambda x: 'good' if x == 1 else 'bad')
    reviews['predictedSentiment'] = reviews['predictedSentiment'].apply(lambda x: 'good' if x == 1 else 'bad')

    # Create plots to explore the data.
    plt.rcParams['figure.figsize'] = (5,5)
    plt.rcParams['font.size'] = 12
    sns.set(rc={'font.size':12,'figure.figsize':(5,5)})
    fig, ax =plt.subplots(1,2)
    plot = sns.countplot(reviews['predictedSentiment'],palette='viridis',ax=ax[0])
    plot.set(xlabel='predicted sentiment')
    plot.get_figure().tight_layout()
    plot = sns.countplot(reviews['actualSentiment'],palette='viridis',ax=ax[1])
    plot.set(xlabel='actual sentiment')
    plot.get_figure().tight_layout()
    # fig = plot.get_figure()
    filename = "static/images/plots/countplot1{}{}{}".format(random.randint(1000,2000),round(time.time()),'.png')
    title = "Predicted Sentiment vs Actual Sentiment"
    description = "This graph shows the count of reviews categorized by predicted sentiment of good or bad. For the purposes of this application, 3-5 stars are considered good and 1-2 stars are considered bad. Within each predicted category, the reviews are further divided between those that actually had good sentiment and those that actually had bad sentiment."
    images.append([filename,title,description])
    # fig.tight_layout()
    plt.savefig(filename)
    plt.tight_layout()
    # buf = io.BytesIO()
    # FigureCanvas(fig).print_png(buf)
    # return Response(buf.getvalue(),mimetype='image/png')
    plt.close()

    plt.rcParams['figure.figsize'] = (5,5)
    plt.rcParams['font.size'] = 12
    sns.set(rc={'font.size':12,'figure.figsize':(5,5)})
    plot = sns.countplot(reviews['overall'])
    plot.set(xlabel='stars')
    fig = plot.get_figure()
    filename = "static/images/plots/countplot{}{}{}".format(random.randint(1000,2000),round(time.time()),'.png')
    title = "Actual Star Rating"
    description = "This graph shows the count of reviews categorized by number of stars."
    images.append([filename,title,description])
    fig.tight_layout()
    fig.savefig(filename)
    # buf = io.BytesIO()
    # FigureCanvas(fig).print_png(buf)
    # return Response(buf.getvalue(),mimetype='image/png')
    plt.close()

    plt.rcParams['figure.figsize'] = (5,5)
    plt.rcParams['font.size'] = 12
    sns.set(rc={'font.size':12,'figure.figsize':(5,5)})
    plot = WordCloud(width=800,height=800,background_color='white',stopwords=set(STOPWORDS),min_font_size=10).generate_from_frequencies(freqdist)
    # output = io.BytesIO()
    # wordcloud.to_image().save(output, format='PNG')
    filename = "static/images/plots/wordcloud{}{}{}".format(random.randint(1000,2000),round(time.time()),'.png')
    title = "Most Frequent Words"
    description = "A word picture of the 40 most frequent words in the dataset."
    images.append([filename,title,description])
    plot.to_image().save(filename, format='PNG')
    # wordcloud = Response(output.getvalue(),mimetype='image/png')
    # return wordcloud
    plt.close()

    plt.rcParams['figure.figsize'] = (5,5)
    plt.rcParams['font.size'] = 12
    sns.set(rc={'font.size':12,'figure.figsize':(5,5)})
    plot = sns.distplot(reviews['length'],kde=False)
    plot.set(xlabel='review length')
    plot.set(ylabel='count')
    fig = plot.get_figure()
    filename = "static/images/plots/distplot{}{}{}".format(random.randint(1000,2000),round(time.time()),'.png')
    title = "Review Length"
    description = "This plot shows the distribution of reviews by the length of the review text (number of characters)."
    images.append([filename,title,description])
    fig.tight_layout()
    fig.savefig(filename)
    plt.close()

    plt.rcParams['figure.figsize'] = (5,5)
    plt.rcParams['font.size'] = 12
    sns.set(rc={'font.size':12,'figure.figsize':(5,5)})
    plot = sns.heatmap(reviews.drop(['predictedSentimentConfidence'],axis=1).corr(),cmap='coolwarm')
    fig = plot.get_figure()
    filename = "static/images/plots/heatmap{}{}{}".format(random.randint(1000,2000),round(time.time()),'.png')
    title = "Variable Correlation Heatmap"
    description = "This plot shows the correlation between the numeric variables in the dataset."
    images.append([filename,title,description])
    fig.tight_layout()
    fig.savefig(filename)
    plt.close()

    plt.rcParams['figure.figsize'] = (5,5)
    plt.rcParams['font.size'] = 12
    sns.set(rc={'font.size':12,'figure.figsize':(5,5)})
    plot = sns.pairplot(reviews.select_dtypes(['number']).drop(['predictedSentimentConfidence'],axis=1).dropna())
    filename = "static/images/plots/pairplot{}{}{}".format(random.randint(1000,2000),round(time.time()),'.png')
    title = "Pair Plot"
    description = "This graph provides a scatterplot plotting each numeric variable against all the others to identify trends."
    images.append([filename,title,description])
    plot.savefig(filename)
    # fig = plot.fig
    # buf = io.BytesIO()
    # FigureCanvas(fig).print_png(buf)
    # return Response(buf.getvalue(),mimetype='image/png')
    plt.close()

    datasource = "The source of the dataset is <a target='_blank' href='https://nijianmo.github.io/amazon/index.html'>https://nijianmo.github.io/amazon/index.html</a>."

    # Create the html and return it to the user.
    accuracy = render_template('accuracy.html', class_report=class_report, conf_matrix=conf_matrix, accuracy=accuracy)
    layout = render_template('choose-analysis.html', datasets=datasets, datasetname=dataset.title(), datasource=datasource, output=render_template('data-exploration.html', accuracy=accuracy, images=images, datasetTable=datasetTable, coefficientTables=coefficientTables))
    return layout

@app.route('/analyze-content/')
@require_login()
def analyzeReview():
    """
    UI form to provide a single review for sentiment analysis.
    """
    return render_template('analyze-review-form.html')

@app.route('/analyze-content/',methods=['POST'])
@require_login()
def predictReview():
    """
    API: Predict the sentiment in a single review.
    """

    X = [clean_review(request.form['review'])]

    mmc = MultiModelClassifier(3)
    prediction, confidence = mmc.predict(X)

    if prediction[0] == 1:
        sentiment = "<span style='color:green; float:center'>{0:.2f}% Positive</span>".format(confidence[0]*100)
    else:
        sentiment = "<span style='color:red'>{0:.2f}% Negative</span>".format(confidence[0]*100)

    return render_template('analyze-review-form.html', output=sentiment)

@app.route('/analyze-file/')
@require_login()
def analyzeFile():
    """
    UI form to provide a file for sentiment analysis.
    """
    return render_template('analyze-file-form.html')

@app.route('/analyze-file/',methods=['POST'])
@require_login()
def predictFile():
    """
    API to predict the sentiment in a csv file and return a table with sentiment and confidence columns appended.
    """

    # Receive a file from the user.
    file = request.files.get('dataFile')

    # Confirm that the file is a csv file.
    if file:
        mimetype = file.content_type
        if mimetype != 'text/csv' and mimetype != 'application/vnd.ms-excel':
            return render_template('analyze-file-form.html', output="<span style='color:red;'>File type must be <b>csv</b>. Type was: {}</span>".format(mimetype))
    else:
        return render_template('analyze-file-form.html', output="<span style='color:red;'>Couldn't read file.</span>")

    # Attempt to read the csv file. Return if the file is empty and read fails.
    try:
        reviews = pd.read_csv(file)
    except:
        return render_template('analyze-file-form.html', output="<span style='color:red;'>File must be a csv file and cannot be empty.</span>")

    # Return if the uploaded file doesn't have the required columns.
    if 'reviewText' not in reviews.columns:
        errorMessage = "<span style='color:red;'>File does not contain a <b>reviewText</b> column.</span>"
        return render_template('analyze-file-form.html', output=errorMessage)

    # Create a frequency distriction of all the words.
    all_words = []
    reviews.dropna(subset=['reviewText'],inplace=True)
    reviews['reviewTextClean'] = reviews['reviewText'].apply(clean_review,all_words=all_words)
    reviews.dropna(subset=['reviewTextClean'],inplace=True)
    reviews['length'] = reviews['reviewText'].apply(len)
    freqdist = nltk.FreqDist(all_words)

    # Check that at least one review had at least one meaningful word
    if len(freqdist) == 0:
        return render_template('analyze-file-form.html', output="<span style='color:red;'>File did not contain any reviews with any meaningful words.</span>")

    # Predict the sentiment in the reviews. Return if the file doesn't contain any rows.
    X = reviews['reviewTextClean']
    mmc = MultiModelClassifier(1)
    try:
        predictions, confidence = mmc.predict(X)
    except:
        return render_template('analyze-file-form.html', output="<span style='color:red;'>File must contain at least one row.</span>")

    coefficientTables = createCoefficientsTables(mmc,freqdist)

    # Append the prediction and confidence tables to the original dataset.
    reviews['predictedSentiment'] = predictions
    reviews['predictedSentimentConfidence'] = confidence

    images = []

    # Create data visualization charts to return to the user.
    plt.rcParams['figure.figsize'] = (5,5)
    plt.rcParams['font.size'] = 12
    sns.set(rc={'font.size':12,'figure.figsize':(5,5)})
    wordcloud = WordCloud(width=800,height=800,background_color='white',stopwords=set(STOPWORDS),min_font_size=10).generate_from_frequencies(freqdist)
    filename = "static/images/plots/wordcloud{}{}{}".format(random.randint(1000,2000),round(time.time()),'.png')
    title = "Most Frequent Words"
    description = "A word picture of the 40 most frequent words in the dataset."
    images.append([filename,title,description])
    wordcloud.to_image().save(filename, format='PNG')
    plt.close()

    plt.rcParams['figure.figsize'] = (5,5)
    plt.rcParams['font.size'] = 12
    sns.set(rc={'font.size':12,'figure.figsize':(5,5)})
    countplot = sns.countplot(reviews['predictedSentiment'])
    countplot.set(xlabel="predicted sentiment")
    fig = countplot.get_figure()
    filename = "static/images/plots/countplot{}{}{}".format(random.randint(1000,2000),round(time.time()),'.png')
    title = "Predicted Sentiment"
    description = "This graph shows the count of reviews categorized by predicted sentiment of positive (1) or negative (0)."
    images.append([filename,title,description])
    fig.tight_layout()
    fig.savefig(filename)
    plt.close()

    plt.rcParams['figure.figsize'] = (5,5)
    plt.rcParams['font.size'] = 12
    sns.set(rc={'font.size':12,'figure.figsize':(5,5)})
    distplot = sns.distplot(reviews['length'],kde=False)
    distplot.set(xlabel="review length")
    distplot.set(ylabel="count")
    fig = distplot.get_figure()
    filename = "static/images/plots/distplot{}{}{}".format(random.randint(1000,2000),round(time.time()),'.png')
    title = "Review Length"
    description = "This plot shows the distribution of reviews by the length of the review text (number of characters)."
    images.append([filename,title,description])
    fig.tight_layout()
    fig.savefig(filename)
    plt.close()

    # The seaborn heatmap requires at least 2 data points. Don't return a heatmap if a file with only 1 row is uploaded.
    if len(freqdist) > 1:
        plt.rcParams['figure.figsize'] = (5,5)
        plt.rcParams['font.size'] = 12
        sns.set(rc={'font.size':12,'figure.figsize':(5,5)})
        heatmap = sns.heatmap(reviews.select_dtypes(['number']).corr(),cmap='coolwarm')
        fig = heatmap.get_figure()
        filename = "static/images/plots/heatmap{}{}{}".format(random.randint(1000,2000),round(time.time()),'.png')
        title = "Variable Correlation Heatmap"
        description = "This plot shows the correlation between the numeric variables in the dataset."
        images.append([filename,title,description])
        fig.tight_layout()
        fig.savefig(filename)
        plt.close()

    plt.rcParams['figure.figsize'] = (5,5)
    plt.rcParams['font.size'] = 12
    sns.set(rc={'font.size':12,'figure.figsize':(5,5)})
    plot = sns.pairplot(reviews.select_dtypes(['number']).drop(['predictedSentimentConfidence'],axis=1).dropna())
    filename = "static/images/plots/pairplot{}{}{}".format(random.randint(1000,2000),round(time.time()),'.png')
    title = "Pair Plot"
    description = "This graph provides a scatterplot plotting each numeric variable against all the others to identify trends."
    images.append([filename,title,description])
    plot.savefig(filename)
    plt.close()

    # Save the file to allow the user to download it if they choose.
    filename = "tempFile{}{}".format(random.randint(0,1000),round(time.time()))
    reviews.to_csv('tempFiles/{}.csv'.format(filename))
    datasetTable = """
    <div>
        <form action="\download-file\{}" method="GET"><button type="submit" class="btn btn-primary">Download Updated Dataset</button></form>
        <br/>
        {}
    </div>
    """.format(filename,reviews.head(10).to_html(classes='table table-striped table-hover table-sm table-responsive'))

    return render_template('analyze-file-form.html', output=render_template('data-exploration.html', images=images, datasetTable=datasetTable, coefficientTables=coefficientTables))

@app.route('/download-file/<filename>',methods=['GET'])
@require_login()
def downloadFile(filename):
    """
    Send the requested file to the user.
    """
    filename = re.sub('[^A-Za-z0-9]', '', filename) # Guard against attempts to gain access to the server
    return send_file('tempFiles/{}.csv'.format(filename), as_attachment=True, attachment_filename="updatedFile.csv")

@app.errorhandler(413)
def request_entity_too_large(e):
    return render_template('analyze-file-form.html', output="<span style='color:red;'>File size cannot be more than 10MB.</span>")

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html', output="<h1>Hi there! You've reached the end of the internet.</h1>")

@app.errorhandler(500)
def server_error(e):
    return render_template('choose-analysis.html', output="<span>An error occurred. Please try again.</span>")

def createCoefficientsTables(mmc,freqdist):
    """
    Create tables for the top 10 most positive and negative coefficients for each of the 4 linear models used by the MultiModelClassifier.
    """

    tables = []

    logisticRegressionCoefficients = mmc.logisticRegressionCoefficients()
    pos_logisticRegressionCoefficients = lastXCoefficients(10,logisticRegressionCoefficients,freqdist)
    neg_logisticRegressionCoefficients = firstXCoefficients(10,logisticRegressionCoefficients,freqdist)
    coefficientList = createCoefficientsList(pos_logisticRegressionCoefficients,neg_logisticRegressionCoefficients)
    table = render_template('table.html',matrix=coefficientList)
    tables.append([table,'Logistic Regression', 'These are the top 10 most positive and top 10 most negative words identified by the Linear Support Vector classification model. A positive coefficient increases likelihood that the predicted target value will be 1 while a negative coefficient increases the likelihood that the predicted target value will be 0.'])

    stochasticGradientCoefficients = mmc.stochasticGradientCoefficients()
    pos_stochasticGradientCoefficients = lastXCoefficients(10,stochasticGradientCoefficients,freqdist)
    neg_stochasticGradientCoefficients = firstXCoefficients(10,stochasticGradientCoefficients,freqdist)
    coefficientList = createCoefficientsList(pos_stochasticGradientCoefficients,neg_stochasticGradientCoefficients)
    table = render_template('table.html',matrix=coefficientList)
    tables.append([table,'Stochastic Gradient Descent', 'These are the top 10 most positive and top 10 most negative words identified by the Stochastic Gradient Descent classification model. A positive coefficient increases likelihood that the predicted target value will be 1 while a negative coefficient increases the likelihood that the predicted target value will be 0.'])

    linearSVCCoefficients = mmc.linearSVCCoefficients()
    pos_linearSVCCoefficients = lastXCoefficients(10,linearSVCCoefficients,freqdist)
    neg_linearSVCCoefficients = firstXCoefficients(10,linearSVCCoefficients,freqdist)
    coefficientList = createCoefficientsList(pos_linearSVCCoefficients,neg_linearSVCCoefficients)
    table = render_template('table.html',matrix=coefficientList)
    tables.append([table, 'Linear Support Vector', 'These are the top 10 most positive and top 10 most negative words identified by the Linear Support Vector classification model. A positive coefficient increases likelihood that the predicted target value will be 1 while a negative coefficient increases the likelihood that the predicted target value will be 0.'])

    table = [['Relevant Words', 'Coefficient', 'Irrelevant Words', 'Coefficient']]
    multinomialNBCoefficients = mmc.multinomialNBCoefficients()
    pos_multinomialNBCoefficients = lastXCoefficients(10,multinomialNBCoefficients,freqdist,ignorePositives=False)
    neg_multinomialNBCoefficients = firstXCoefficients(10,multinomialNBCoefficients,freqdist,ignoreNegatives=False)
    coefficientList = createCoefficientsList(pos_multinomialNBCoefficients,neg_multinomialNBCoefficients)
    coefficientList[0][0] = 'Relevant Words'
    coefficientList[0][2] = 'Irrelevant Words'
    table = render_template('table.html',matrix=coefficientList)
    tables.append([table, 'Multinomial Naive Bayes', 'These are the top 10 most relevant and irrelevant words in respect to a review being classified as positive as identified by the Multinomial Naive Bayes model. Relevant words increase the likelihood of the review being classified as positive while irrelevant words have little impact on the classification.'])

    return tables

def createCoefficientsList(positiveCoefficientList,negativeCoefficientList):
    """
    Combine a list of positive and negative coefficients into a single list.
    """

    list = [['Positive Words', 'Coefficient', 'Negative Words', 'Coefficient']]

    for i in range(max(len(positiveCoefficientList),len(negativeCoefficientList))):
        row = []
        if len(positiveCoefficientList) > i:
            row.append(positiveCoefficientList[i][0])
            row.append(positiveCoefficientList[i][1])
        else:
            row.append("")
            row.append("")
        if len(negativeCoefficientList) > i:
            row.append(negativeCoefficientList[i][0])
            row.append(negativeCoefficientList[i][1])
        else:
            row.append("")
            row.append("")

        list.append(row)

    return list

def firstXCoefficients(X, coefTupleList, frequencyDistribution, ignoreNegatives=True):
    """
    Identify the first x words in the coefTupleListt that also appear in
    the frequencyDistribution, starting from the front to the back of the list.
    By default, stop if a positive value is found.
    """
    coefficients = []
    for i in range(len(coefTupleList)):
        if coefTupleList[i][1] in frequencyDistribution:
            if coefTupleList[i][0] > 0 and ignoreNegatives == True:
                return coefficients
            coefficients.append([coefTupleList[i][1],round(coefTupleList[i][0],2)])
            if len(coefficients) == 10:
                return coefficients

    # Return the coefficients if there are less than 10
    return coefficients

def lastXCoefficients(X, coefTupleList, frequencyDistribution, ignorePositives=True):
    """
    Identify the last x words in the coefTupleListt that also appear in
    the frequencyDistribution, starting from the back to the front of the list.
    By default, stop if a negative value is found.
    """
    coefficients = []
    for i in range(len(coefTupleList)-1,-1,-1):
        if coefTupleList[i][1] in frequencyDistribution:
            if coefTupleList[i][0] < 0 and ignorePositives == True:
                return coefficients
            coefficients.append([coefTupleList[i][1],round(coefTupleList[i][0],2)])
            if len(coefficients) == 10:
                return coefficients

    # Return the coefficients if there are less than 10
    return coefficients

def createTableFromClassificationReport(class_report):
    """
    Create a table to display a classification_report.
    """
    cr = class_report.split()
    classes = []
    classes.append([cr[4],cr[5],cr[6],cr[7],cr[8]])
    classes.append([cr[9],cr[10],cr[11],cr[12],cr[13]])
    return render_template('classification-report-table.html', classes=classes)

def createTableFromConfusionMatrix(conf_matrix):
    """
    Create a table to show a confusion_matrix.
    """
    return render_template('confusion-matrix-table.html', conf_matrix=conf_matrix)

def clean_review(text,all_words=None):
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
    if all_words != None:
        [all_words.append(word) for word in words]
    return ' '.join(words)

if __name__ == '__main__':
    stopwords_set = set(nltk.corpus.stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    app.run(debug=False)
