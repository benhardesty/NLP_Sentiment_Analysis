from flask import Flask, request, jsonify, render_template, send_file, session, redirect, url_for, Response
from functools import wraps
import os
import jinja2
import sys
import io
import base64
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def require_login():
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            if not session.get('logged_in'):
                return redirect(url_for('login'))
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
    jumboTitle="Amazon Product Reviews"
    jumboSubTitle="Sentiment Analysis and Data Exploration of amazon product reviews."
    return render_template('choose-analysis.html')

@app.route('/login/')
def login():
    """
    UI for login.
    """
    return render_template('login.html')

@app.route('/login/',methods=['POST'])
def authenticate():
    """
    """
    # return render_template('login.html', output="you rock")
    if request.form['username'] == 'username' and request.form['password'] == 'password':
        session['logged_in'] = True
        return redirect(url_for('home'))

    output = '<label class="form-check-label" for="exampleCheck1" style="color:red;">Login was unsuccessful</label>'
    return render_template('login.html', output=output)

@app.route('/logout/')
@require_login()
def logout():
    session['logged_in'] = False
    return redirect(url_for('home'))

@app.route('/data-exploration/<dataset>')
def analyze_dataset(dataset):
    """

    """
    images = []
    reviews = pd.read_csv('data/{}.csv'.format(dataset),index_col=0)

    # Clean the reviews and get a frequency distribution of all the words.
    all_words = []
    reviews.dropna(subset=['reviewText'],inplace=True)
    reviews['reviewTextClean'] = reviews['reviewText'].apply(clean_review, all_words=all_words)
    reviews.dropna(subset=['reviewTextClean'],inplace=True)
    freqdist = nltk.FreqDist(all_words)

    # Create a sentiment column
    reviews['sentiment'] = reviews['overall'].apply(lambda x: 1 if x >= 3 else 0)

    # Create a column for the length of the reviews.
    reviews['length'] = reviews['reviewText'].apply(len)

    X = reviews['reviewTextClean']

    # Predict the sentiment in the reviews.
    mmc = MultiModelClassifier(1)
    predictions, confidence = mmc.predict(X)

    # Create accuracy reports
    conf_matrix = confusion_matrix(reviews['sentiment'],predictions)
    conf_matrix = createTableFromConfusionMatrix(conf_matrix)
    class_report = classification_report(reviews['sentiment'],predictions)
    class_report = createTableFromClassificationReport(class_report)
    accuracy = "{}%".format(round(accuracy_score(reviews['sentiment'],predictions)*100,2))

    # Create tables explaining the coefficients of the linear models used by the MultiModelClassifier
    coefficientTables = createCoefficientsTables(mmc,freqdist)

    # Add sentiment and confidence to the output data.
    reviews['predictedSentiment'] = predictions
    reviews['predictedSentimentConfidence'] = confidence

    # Create plots to explore the data.
    plt.rcParams['figure.figsize'] = (5,5)
    plt.rcParams['font.size'] = 12
    sns.set(rc={'font.size':12,'figure.figsize':(5,5)})

    plot = WordCloud(width=800,height=800,background_color='white',stopwords=set(STOPWORDS),min_font_size=10).generate_from_frequencies(freqdist)
    # output = io.BytesIO()
    # wordcloud.to_image().save(output, format='PNG')
    filename = "static/images/plots/wordcloud{}{}{}".format(random.randint(1000,2000),round(time.time()),'.png')
    title = "Word Cloud"
    description = "A word picture of the 40 most frequent words in the dataset."
    images.append([filename,title,description])
    plot.to_image().save(filename, format='PNG')
    # wordcloud = Response(output.getvalue(),mimetype='image/png')
    # return wordcloud
    plt.close()

    plot = sns.countplot(reviews['overall'])
    fig = plot.get_figure()
    filename = "static/images/plots/countplot{}{}{}".format(random.randint(1000,2000),round(time.time()),'.png')
    title = "Count Plot"
    description = "This graph shows the count of reviews categorized by number of stars."
    images.append([filename,title,description])
    fig.tight_layout()
    fig.savefig(filename)
    # buf = io.BytesIO()
    # FigureCanvas(fig).print_png(buf)
    # return Response(buf.getvalue(),mimetype='image/png')
    plt.close()

    plot = sns.distplot(reviews['length'],kde=False)
    fig = plot.get_figure()
    filename = "static/images/plots/distplot{}{}{}".format(random.randint(1000,2000),round(time.time()),'.png')
    title = "Distribution Plot"
    description = "This plot shows the distribution of reviews by the length of the review text."
    images.append([filename,title,description])
    fig.tight_layout()
    fig.savefig(filename)
    plt.close()

    plot = sns.heatmap(reviews.drop(['predictedSentimentConfidence'],axis=1).corr(),cmap='coolwarm')
    fig = plot.get_figure()
    filename = "static/images/plots/heatmap{}{}{}".format(random.randint(1000,2000),round(time.time()),'.png')
    title = "Heatmap"
    description = "This plot shows the correlation between the numeric variables in the dataset."
    images.append([filename,title,description])
    fig.tight_layout()
    fig.savefig(filename)
    plt.close()

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

    # Save the temporary file.
    # reviews.to_csv('tempFiles/tempFile.csv')

    datasource = "The source of the datasets is <a target='_blank' href='https://nijianmo.github.io/amazon/index.html'>https://nijianmo.github.io/amazon/index.html</a>"
    datasetTable = reviews.head(10).to_html(classes='table table-striped table-hover table-sm table-responsive')
    accuracy = render_template('accuracy.html', class_report=class_report, conf_matrix=conf_matrix, accuracy=accuracy)
    layout = render_template('choose-analysis.html', datasetname=dataset.capitalize(), datasource=datasource, output=render_template('data-exploration.html', accuracy=accuracy, images=images, datasetTable=datasetTable, coefficientTables=coefficientTables))

    return layout

@app.route('/analyze-review/')
@require_login()
def analyzeReview():
    """
    UI form to provide a single review for sentiment analysis.
    """
    return render_template('analyze-review-form.html')

@app.route('/analyze-review/',methods=['POST'])
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

    file = request.files.get('dataFile')

    if file:
        mimetype = file.content_type
        if mimetype != 'text/csv':
            return render_template('analyze-file-form.html', output="<span style='color:red;'>File type must be <b>csv</b>.</span>")
    else:
        return render_template('analyze-file-form.html', output="<span style='color:red;'>Couldn't read file.</span>")

    reviews = pd.read_csv(file)

    if 'reviewText' not in reviews.columns:
        errorMessage = "<span style='color:red;'>File does not contain a <b>reviewText</b> column.</span>"
        return render_template('analyze-file-form.html', output=errorMessage)

    all_words = []
    reviews.dropna(subset=['reviewText'],inplace=True)
    reviews['reviewTextClean'] = reviews['reviewText'].apply(clean_review,all_words=all_words)
    reviews.dropna(subset=['reviewTextClean'],inplace=True)
    reviews['length'] = reviews['reviewText'].apply(len)
    freqdist = nltk.FreqDist(all_words)

    X = reviews['reviewTextClean']

    mmc = MultiModelClassifier(1)

    predictions, confidence = mmc.predict(X)

    coefficientTables = createCoefficientsTables(mmc,freqdist)

    reviews['predictedSentiment'] = predictions
    reviews['predictedSentimentConfidence'] = confidence
    reviews.drop(['reviewTextClean'],axis=1,inplace=True)

    plt.rcParams['figure.figsize'] = (5,5)
    plt.rcParams['font.size'] = 12
    sns.set(rc={'font.size':12,'figure.figsize':(5,5)})

    images = []

    wordcloud = WordCloud(width=800,height=800,background_color='white',stopwords=set(STOPWORDS),min_font_size=10).generate_from_frequencies(freqdist)
    filename = "static/images/plots/wordcloud{}{}{}".format(random.randint(1000,2000),round(time.time()),'.png')
    title = "Word Cloud"
    description = "A word picture of the 40 most frequent words in the dataset."
    images.append([filename,title,description])
    wordcloud.to_image().save(filename, format='PNG')
    plt.clf()

    countplot = sns.countplot(reviews['predictedSentiment'])
    fig = countplot.get_figure()
    filename = "static/images/plots/countplot{}{}{}".format(random.randint(1000,2000),round(time.time()),'.png')
    title = "Count Plot"
    description = "This graph shows the count of reviews categorized by predicted sentiment of positive (1) or negative (0)."
    images.append([filename,title,description])
    fig.tight_layout()
    fig.savefig(filename)
    plt.clf()

    distplot = sns.distplot(reviews['length'],kde=False)
    fig = distplot.get_figure()
    filename = "static/images/plots/distplot{}{}{}".format(random.randint(1000,2000),round(time.time()),'.png')
    title = "Distribution Plot"
    description = "This plot shows the distribution of reviews by the length of the review text."
    images.append([filename,title,description])
    fig.tight_layout()
    fig.savefig(filename)
    plt.clf()

    heatmap = sns.heatmap(reviews.select_dtypes(['number']).corr(),cmap='coolwarm')
    fig = heatmap.get_figure()
    filename = "static/images/plots/heatmap{}{}{}".format(random.randint(1000,2000),round(time.time()),'.png')
    title = "Heatmap"
    description = "This plot shows the correlation between the numeric variables in the dataset."
    images.append([filename,title,description])
    fig.tight_layout()
    fig.savefig(filename)
    plt.clf()

    plot = sns.pairplot(reviews.select_dtypes(['number']).drop(['predictedSentimentConfidence'],axis=1).dropna())
    filename = "static/images/plots/pairplot{}{}{}".format(random.randint(1000,2000),round(time.time()),'.png')
    title = "Pair Plot"
    description = "This graph provides a scatterplot plotting each numeric variable against all the others to identify trends."
    images.append([filename,title,description])
    plot.savefig(filename)
    plt.clf()

    filename = "tempFile{}{}".format(random.randint(0,1000),round(time.time()))

    # Save the temporary file.
    # reviews.to_csv('tempFiles/tempFile.csv')
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
    Create tables for the top 10 positive and negative coefficients for each of the 4 linear models used by the MultiModelClassifier.
    """
    tables = []

    linearSVCCoefficients = mmc.linearSVCCoefficients()
    pos_linearSVCCoefficients = [['Word','Coefficient']] + lastXCoefficients(10,linearSVCCoefficients,freqdist)
    neg_linearSVCCoefficients = [['Word','Coefficient']] + firstXCoefficients(10,linearSVCCoefficients,freqdist)
    pos_linearSVCCoefficients = render_template('table.html',matrix=pos_linearSVCCoefficients)
    neg_linearSVCCoefficients = render_template('table.html',matrix=neg_linearSVCCoefficients)

    tables.append([pos_linearSVCCoefficients,
                    'LinearSVC Coefficients - Positive',
                    'These are the top 10 most positive words. A positive coefficient increases the resulting target value and increases the likelihood of the review being classified as positive.'])
    tables.append([neg_linearSVCCoefficients,
                    'LinearSVC Coefficients - Negative',
                    'These are the top 10 most negative words. A negative coefficient decreases the resulting target value and increases the likelihood of the review being classified as negative.'])

    logisticRegressionCoefficients = mmc.logisticRegressionCoefficients()
    pos_logisticRegressionCoefficients = [['Word','Coefficient']] + lastXCoefficients(10,logisticRegressionCoefficients,freqdist)
    neg_logisticRegressionCoefficients = [['Word','Coefficient']] + firstXCoefficients(10,logisticRegressionCoefficients,freqdist)
    pos_logisticRegressionCoefficients = render_template('table.html',matrix=pos_logisticRegressionCoefficients)
    neg_logisticRegressionCoefficients = render_template('table.html',matrix=neg_logisticRegressionCoefficients)
    tables.append([pos_logisticRegressionCoefficients,
                    'Logistic Regression Coefficients - Positive',
                    'These are the top 10 most positive words. A positive coefficient increases the resulting target value and increases the likelihood of the review being classified as positive.'])
    tables.append([neg_logisticRegressionCoefficients,
                    'Logistic Regression Coefficients - Negative',
                    'These are the top 10 most negative words. A negative coefficient decreases the resulting target value and increases the likelihood of the review being classified as negative.'])

    stochasticGradientCoefficients = mmc.stochasticGradientCoefficients()
    pos_stochasticGradientCoefficients = [['Word','Coefficient']] + lastXCoefficients(10,stochasticGradientCoefficients,freqdist)
    neg_stochasticGradientCoefficients = [['Word','Coefficient']] + firstXCoefficients(10,stochasticGradientCoefficients,freqdist)
    pos_stochasticGradientCoefficients = render_template('table.html',matrix=pos_stochasticGradientCoefficients)
    neg_stochasticGradientCoefficients = render_template('table.html',matrix=neg_stochasticGradientCoefficients)
    tables.append([pos_stochasticGradientCoefficients,
                    'Stochastic Gradient Coefficients - Positive',
                    'These are the top 10 most positive words. A positive coefficient increases the resulting target value and increases the likelihood of the review being classified as positive.'])
    tables.append([neg_stochasticGradientCoefficients,
                    'Stochastic Gradient Coefficients - Negative',
                    'These are the top 10 most negative words. A negative coefficient decreases the resulting target value and increases the likelihood of the review being classified as negative.'])

    multinomialNBCoefficients = mmc.multinomialNBCoefficients()
    pos_multinomialNBCoefficients = [['Word','Coefficient']] + lastXCoefficients(10,multinomialNBCoefficients,freqdist)
    pos_multinomialNBCoefficients = render_template('table.html',matrix=pos_multinomialNBCoefficients)
    tables.append([pos_multinomialNBCoefficients,
                    'Multinomial Naive Bayes - Relative to Positive Sentiment',
                    'These are the top 10 most relevant/positive words. The greater the coefficient, the greater the target value is increased, which increases the impact the word has on the review being classified as positive.'])
    neg_multinomialNBCoefficients = [['Word','Coefficient']] + firstXCoefficients(10,multinomialNBCoefficients,freqdist)
    neg_multinomialNBCoefficients = render_template('table.html',matrix=neg_multinomialNBCoefficients)
    tables.append([neg_multinomialNBCoefficients,
                    'Multinomial Naive Bayes - Irrelative to Positive Sentiment',
                    'These are the top 10 least relevant/positive words. The smaller the coefficient, the less the target value is increased, which reduces the impact the word has on the review being classified as positive.'])
    return tables

def firstXCoefficients(X, coefTupleList, frequencyDistribution):
    """
    Identify the first 10 words in the reviews dataset that also appear in
    the coefficient list, starting from the front to the back of the list.
    """
    coefficients = []
    for i in range(len(coefTupleList)):
        if coefTupleList[i][1] in frequencyDistribution:
            coefficients.append([coefTupleList[i][1],coefTupleList[i][0]])
            if len(coefficients) == 10:
                return coefficients

def lastXCoefficients(X, coefTupleList, frequencyDistribution):
    """
    Identify the first 10 words in the reviews dataset that also appear in
    the coefficient list, starting from the back to the front of the list.
    """
    coefficients = []
    for i in range(len(coefTupleList)-1,-1,-1):
        if coefTupleList[i][1] in frequencyDistribution:
            coefficients.append([coefTupleList[i][1],coefTupleList[i][0]])
            if len(coefficients) == 10:
                return coefficients

def createTableFromClassificationReport(class_report):
    """
    Create a table to display a classification_report.
    """
    cr = class_report.split()
    table = """
    <table class="table table-hover">
        <tr>
            <td></td>
            <th>precision</th>
            <th>recall</th>
            <th>f1-score</th>
            <th>support</th>
        </tr>
        <tr>
            <th>{}</th>
            <td>{}</td>
            <td>{}</td>
            <td>{}</td>
            <td>{}</td>
        </tr>
        <tr>
            <th>{}</th>
            <td>{}</td>
            <td>{}</td>
            <td>{}</td>
            <td>{}</td>
        </tr>
    </table>
    """.format(cr[4],cr[5],cr[6],cr[7],cr[8],cr[9],cr[10],cr[11],cr[12],cr[13])
    return table

def createTableFromConfusionMatrix(conf_matrix):
    """
    Create a table to show a confusion_matrix.
    """
    table = """
    <table class="table table-hover">
        <tr>
            <td></td>
            <td></td>
            <th colspan="2">Predicted Sentiment</th>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <th>Negative</th>
            <th>Positive</th>
        </tr>
        <tr>
            <th rowspan="3" style="text-align:center;">Actual Sentiment</th>
        </tr>
        <tr>
            <th>Negative</th>
            <td>{}</td>
            <td>{}</td>
        </tr>
        <tr>
            <th>Positive</th>
            <td>{}</td>
            <td>{}</td>
        </tr>
    </table>
    """.format(conf_matrix[0][0],conf_matrix[0][1],conf_matrix[1][0],conf_matrix[1][1])
    return table

def clean_review(text,all_words=None):
    """
    Clean a review so it can be used for ML.
    1. Remove punctuation.
    2. Put all letters to lowercase.
    3. Split the string into an array of words.
    4. Lemmatize each word.
    5. Join the words back together and return the string.
    """

    text = re.sub('[^A-Za-z]', ' ', text) # Remove all characters other than letters.
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stopwords_set]
    text = [lemmatizer.lemmatize(word) for word in text]
    if all_words != None:
        [all_words.append(word) for word in text]
    return ' '.join(text)

if __name__ == '__main__':
    stopwords_set = set(nltk.corpus.stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    app.run(debug=True)
