# Sentiment-Analysis-regarding-Covid-19

Yating Liang, Yachen Li, George Sun

## Project Summary:
In this project, we are aimed to conduct a sentiment analysis of tweets about 
COVID-19 by creating machine learning models to fit the training set, such as TextBlob, 
Vader, Naive Bayes, Logistic Regression, Random Forest and Neural Network.
Through the analysis we hope to find the best model which can produce the highest 
metrics' values, so that this model can be preserved for further prediction of COVID-19
tweets. In the python file, we tried two datasets, one with 5 categories of sentiment labels 
and the other one with just 3 categories of sentiment labels. Through the comparison of 
the performance of the models on these 2 datasets, we can tend to find which dataset is more 
suitable to train the models.

## Code Execution:
Please navigate to this directory on terminal and type:

`python main.py --test_file Corona_NLP_test.csv --train_file Corona_NLP_train.csv`


If you cannot successfully run the neural network model please install:
`Scikt-Learn` and `Tensorflow 2.0`.

Here is the command line code you need to install tensorflow:

<pre><code>pip install --upgrade tensorflow
</code></pre>


## Steps in main.py:
### 1. Data preprocessing:
Remove stopwords, remove url, remove punctuation and numbers, 
transform to lower cases, lemmatize, and delete useless columns.

### 2. CountVectorizer
Texts were transformed into count vectorizer for some models, such as Logistic regression, Naive bayes, and etc.

### 3. Apply models and evaluate the metrics:
Apply Naive Bayes, Logistic Regression and Random Forest. Then
calculate F1, precision, recall, accuracy.

### 4. Create new dataframe
Substitute extra labels such as "Extremely Positive" and "Extremely Negative"
with "positive" and "negative".

### 5. Methodology
Apply six models on the new dataset, TextBlob, Vader, Logistic Regression,
Naive Bayes, Random Forest and Neural Network.

### 6. Comparison
Treat TextBlob as the baseline method and compare its metrics with the metrics
from the other methods.


## Source of dataset:
https://www.kaggle.com/datatattle/covid-19-nlp-text-classification


