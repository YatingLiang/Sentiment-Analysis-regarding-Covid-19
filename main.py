#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 00:05:07 2021

@author: liangyating
"""
import argparse
import pandas as pd
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import re
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
nltk.download('vader_lexicon')
import warnings
warnings.filterwarnings("ignore")


#preprocessing
def preprocessing_data(text):
    lemma=WordNetLemmatizer()
    text=re.sub(r'http\S+',' ',text) #remove url
    text=re.sub('[^a-z-A-Z]',' ',text) #remove punctuation and numbers
    text=str(text).lower()   #lower cases
    text=text.split()
    text=" ".join([lemma.lemmatize(item) for item in text]) #lemmatize
    return text

def encoding(x):
    encoder = preprocessing.LabelEncoder()
    x_encoded = encoder.fit_transform(x)
    return x_encoded


def naive_bayes(X_train, y_train, X_test, y_test):
    nb = MultinomialNB().fit(X_train, y_train)
    predictions = nb.predict(X_test)
    acc1 = accuracy_score(y_test, predictions)
    # macro calculates metrics for each label, and find their unweighted mean.
    f11 = f1_score(y_test, predictions, average='macro')
    p1 = precision_score(y_test, predictions, average='macro')
    r1 = recall_score(y_test, predictions, average='macro')
    print("Applying Naive Bayes Model:")
    print("Accuracy = {}".format(acc1))
    print("Precision = {}".format(p1))
    print("Recall = {}".format(r1))
    print("F1 = {}".format(f11))
    metrics = [acc1,f11,p1,r1]
    return metrics    
    
def logistic_regression(X_train, y_train, X_test, y_test):
    logreg = LogisticRegression(solver='lbfgs', max_iter=1000).fit(X_train, y_train)
    predictions = logreg.predict(X_test)
    acc2 = accuracy_score(y_test, predictions)
    f12 = f1_score(y_test, predictions, average='macro')
    p2 = precision_score(y_test, predictions, average='macro')
    r2 = recall_score(y_test, predictions, average='macro')
    print("Applying Logistic Regression Model:")
    print("Accuracy = {}".format(acc2))
    print("F1 = {}".format(f12))
    print("Precision = {}".format(p2))
    print("Recall = {}".format(r2))
    
    metrics = [acc2,f12,p2,r2]
    return metrics   

def random_forest(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=42)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    acc3 = accuracy_score(y_test, predictions)
    f13 = f1_score(y_test, predictions, average='macro')
    p3 = precision_score(y_test, predictions, average='macro')
    r3 = recall_score(y_test, predictions, average='macro')
    print("Applying Random Forest Model:")
    print("Accuracy = {}".format(acc3))
    print("F1 = {}".format(f13))
    print("Precision = {}".format(p3))
    print("Recall = {}".format(r3))
   
    metrics = [acc3,f13,p3,r3]
    return metrics

def vader_preprocess(X_test):
    sia = SentimentIntensityAnalyzer()
    score = X_test.apply(lambda X_test: sia.polarity_scores(X_test))
    compound  = score.apply(lambda score_dict: score_dict['compound'])
    return compound
    

def vader(X_train, y_train, X_test, y_test):
    test_compound = vader_preprocess(X_test)
    vader = pd.DataFrame(np.arange(0).reshape(len(X_test),0))
    vader['pred']=np.nan
    for i in range(len(y_test)):
      if test_compound[i] >= 0.05:
        vader['pred'][i] = 'Positive'
      elif test_compound[i] <= -0.05:
        vader['pred'][i] = 'Negative'
      else:
        vader['pred'][i] = 'Neutral'
    acc_v = accuracy_score(y_test, vader['pred'])
    p_v = precision_score(y_test, vader['pred'], average = 'macro')
    r_v = recall_score(y_test, vader['pred'], average = 'macro')
    f1_v = f1_score(y_test, vader['pred'], average = 'macro')
    print("Applying VADER Model:")
    print("Accuracy = {}".format(acc_v))
    print("F1 = {}".format(f1_v))
    print("Precision = {}".format(p_v))
    print("Recall = {}".format(r_v))
    
    metrics = [acc_v,p_v,r_v,f1_v]
    return metrics    
    
def textblob_model(X_train, y_train, X_test, y_test):
    polarity = X_test.apply(lambda X_test: TextBlob(X_test).polarity)
    tb = pd.DataFrame(np.arange(0).reshape(len(X_test),0))
    tb['pred']=np.nan
    for i in range(len(y_test)):
      if polarity[i] > 0.0:
          tb['pred'][i] = 'Positive'
      elif polarity[i] < 0.0:
          tb['pred'][i] = 'Negative'
      else:
          tb['pred'][i] = 'Neutral'
    acc_tb = accuracy_score(y_test, tb['pred'])
    f1_tb = f1_score(y_test, tb['pred'], average = 'macro')
    p_tb = precision_score(y_test, tb['pred'], average = 'macro')
    r_tb = recall_score(y_test, tb['pred'], average = 'macro')
    
    print("Applying TextBlob Model:")
    print("Accuracy = {}".format(acc_tb))
    print("F1 = {}".format(f1_tb))
    print("Precision = {}".format(p_tb))
    print("Recall = {}".format(r_tb))

    metrics = [acc_tb,p_tb,r_tb,f1_tb]
    return metrics    

def LSTM_preprocess(X_train, y_train, X_test, y_test):
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import pandas as pd
    import numpy as np
    y_train_dum = pd.get_dummies(y_train)
    y_test_dum = pd.get_dummies(y_test)
    #train_text = X_train.to_numpy()
    train_text = np.array(X_train)
    #validation_text = X_test.to_numpy()
    validation_text = np.array(X_test)
    #train_labels = y_train_dum.to_numpy()
    train_labels = np.array(y_train_dum)
    #validation_labels = y_test_dum.to_numpy()
    validation_labels = np.array(y_test_dum)
    X = pd.concat([X_train,X_test],axis=0)
    sum_length_of_tweet = 0
    for i in X:
        temp = i
        sum_length_of_tweet = sum_length_of_tweet + len(temp.split())
    max_length = round(sum_length_of_tweet/(len(X)))
    tokenizer = Tokenizer(num_words=10000,oov_token='</OOV>')
    tokenizer.fit_on_texts(train_text)
    train_text_sequences = tokenizer.texts_to_sequences(train_text)
    train_text_padded = pad_sequences(train_text_sequences, maxlen = max_length, padding = 'post')
    validation_text_sequences = tokenizer.texts_to_sequences(validation_text)
    validation_text_padded = pad_sequences(validation_text_sequences, maxlen = max_length, padding = 'post')
    return max_length, train_text_padded, train_labels, validation_text_padded, validation_labels

def f1(y_true, y_pred):
    
    from tensorflow.keras import backend as K
    def recall(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def LSTM_model(X_train, y_train, X_test, y_test):
    import tensorflow as tf
    max_length, train_text_padded, train_labels, validation_text_padded, validation_labels = LSTM_preprocess(X_train, y_train, X_test, y_test)
    print("Neural Network LSTM Model:")
    model_3=tf.keras.models.Sequential([
    tf.keras.layers.Embedding(10000,128,input_length=max_length),
    tf.keras.layers.LSTM(8),
    tf.keras.layers.Dense(8,activation='relu'),
    tf.keras.layers.Dense(3,activation='softmax')
])
    model_3.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=['accuracy',f1, tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
    history_3=model_3.fit(train_text_padded,
                          train_labels,
                          epochs=6,
                          validation_data=(validation_text_padded,validation_labels))
    #model_3.evaluate(validation_text_padded,validation_labels)

    lstm_metrics = list()
    lstm_metrics = model_3.evaluate(validation_text_padded,validation_labels)[1:5]
    print("Applying LSTM Model:")
    print("Accuracy = {}".format(lstm_metrics[0]))
    print("F1 = {}".format(lstm_metrics[1]))
    print("Precision = {}".format(lstm_metrics[2]))
    print("Recall = {}".format(lstm_metrics[3]))
    
    return lstm_metrics

def main(train_file, test_file):
    #import data
    df_test = pd.read_csv(test_file)
    df_train = pd.read_csv(train_file,encoding = 'ISO-8859-1')
    
    #preprocessing data
    #drop useless columns
    df_train.drop(columns=['TweetAt','UserName','ScreenName','Location'],axis=1,inplace=True)
    df_test.drop(columns=['TweetAt','UserName','ScreenName','Location'],axis=1,inplace=True)
    #remove stopwords
    stopword_lst=stopwords.words('english')
    df_train["OriginalTweet"]=df_train["OriginalTweet"].apply(lambda x: " ".join(x for x in x.split() if x not in stopword_lst))
    df_test["OriginalTweet"]=df_test["OriginalTweet"].apply(lambda x: " ".join(x for x in x.split() if x not in stopword_lst))
    #preprocessing
    df_train["OriginalTweet"]=df_train["OriginalTweet"].apply(lambda x : preprocessing_data(x))
    df_test["OriginalTweet"]=df_test["OriginalTweet"].apply(lambda x : preprocessing_data(x))
    
    #train and test data
    X_train = df_train["OriginalTweet"]
    y_train = df_train["Sentiment"]
    X_test = df_test["OriginalTweet"]
    y_test = df_test["Sentiment"]
    
    #encoding
    y_train_encoded = encoding(y_train)
    y_test_encoded = encoding(y_test)
    #count vectorizing
    vectorizer = CountVectorizer()
    vectorizer.fit(X_train)
    X_train_count = vectorizer.transform(X_train)
    X_test_count = vectorizer.transform(X_test)
    
    #apply models
    print("Using 5 labels to train:")
    nb_metrics_5 = naive_bayes(X_train_count, y_train_encoded, X_test_count, y_test_encoded)
    lg_metrics_5 = logistic_regression(X_train_count, y_train_encoded, X_test_count, y_test_encoded)
    rf_metrics_5 =random_forest(X_train_count, y_train_encoded, X_test_count, y_test_encoded)
    
    #combine 5 labels to 3
    y_train_new = df_train["Sentiment"].replace({'Extremely Positive': 'Positive','Extremely Negative': 'Negative'},inplace=False)
    y_test_new = df_test["Sentiment"].replace({'Extremely Positive': 'Positive','Extremely Negative': 'Negative'},inplace=False)
    
    y_train_new_encoded = encoding(y_train_new)
    y_test_new_encoded = encoding(y_test_new)
    
    print("---------------------------")
    print("Using 3 labels to train:")
    tb_metrics_3 = textblob_model(X_train, y_train_new, X_test, y_test_new)
    vader_metrics_3 = vader(X_train, y_train_new, X_test, y_test_new)
    nb_metrics_3 =naive_bayes(X_train_count, y_train_new_encoded, X_test_count, y_test_new_encoded)
    lg_metrics_3 = logistic_regression(X_train_count, y_train_new_encoded, X_test_count, y_test_new_encoded)
    rf_metrics_3 = random_forest(X_train_count, y_train_new_encoded, X_test_count, y_test_new_encoded)
    lstm_metrics_3 = LSTM_model(X_train, y_train_new, X_test, y_test_new)    
    

    
    metrics_df_5 = pd.DataFrame(np.array([nb_metrics_5, lg_metrics_5, rf_metrics_5]),
                       columns=['Accuracy', 'F1', 'Precision', 'Recall'])
    metrics_df_5.index = ['Naive Bayes', 'Logistic Regression', 'Random Forest']
    print("---------------------------")
    print('Result metrics for 5 labels:')
    print(metrics_df_5)
    
    metrics_df_3 = pd.DataFrame(np.array([tb_metrics_3,vader_metrics_3,nb_metrics_3, lg_metrics_3, rf_metrics_3,lstm_metrics_3]),
                       columns=['Accuracy', 'F1', 'Precision', 'Recall'])
    metrics_df_3.index = ['TextBlob', 'Vader','Naive Bayes', 'Logistic Regression', 'Random Forest','Neural Network']
    print("---------------------------")
    print('Result metrics for 3 labels:')
    print(metrics_df_3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, default="./Corona_NLP_test.csv",
                        help="test file")
    parser.add_argument("--train_file", type=str, default="./Corona_NLP_train.csv",
                        help="train file")
    args = parser.parse_args()
    main(args.train_file, args.test_file)
