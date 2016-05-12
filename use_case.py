import time
import json
import logging
import math
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score


logging.captureWarnings(True)

with open('g2.txt') as datafile:
    data=json.load(datafile)
count_vectorizer = CountVectorizer()
jdata=json.dumps(data)
pdata=pd.read_json(jdata)


train,test=train_test_split(data, train_size=0.99)

trainpdata=pd.read_json(json.dumps(train))
testpdata=pd.read_json(json.dumps(test))

tokenize = CountVectorizer().build_tokenizer()

def features(d):
    termsText = tokenize(d['reviewText'])
    termsSummary = tokenize(d['summary'])
    data = {'overall': float(d['overall']), 'unixReviewTime': float(d['unixReviewTime'])}
    data['count']=float(d['count'])
    data['text_length']=len(termsText)
    data['summary_length']=len(termsSummary)
    for t in termsText:
        data[t] = data.get(t, 0) + 1
    for t in termsSummary:
        data[t] = data.get(t, 0) + 1
    return data

vect = DictVectorizer()

X_train = vect.fit_transform(features(d) for d in train)

X_test=vect.transform(features(d) for d in test)

train_targets = trainpdata['hfactor'].values

test_targets = testpdata['hfactor'].values
test={}
##test['overall']=5.0
##test['unixReviewTime']=math.floor(time.time())
##test['reviewText']='I love KIND bars, there are a huge variety of flavours and this is my favorite. I personally use it as a meal replacement and take it along with a Natural Appetite Suppressant(extracted from a fruit). I personally love the taste of dark chocolate. Dark chocolate is an anti-oxidant and provides a variety of health benefits. The bar is very low in sugar so its an ideal meal replacement. It has a good amount of fiber and will help keep your digestive system healthy! The chocolate is smooth and the sweet/saltiness are perfectly balanced. The texture is very crunchy and the flavour releases quickly. I have them as a snack or replace my morning meal. Its a perfect natural snack bar.'
##test['summary']='Tastes amazing!'
##test['count']=4200
test['overall']=float(input('Enter overall product rating\n'))
test['unixReviewTime']=math.floor(time.time())
test['reviewText']=input('Enter the review text enclosed in quotes\n')
test['summary']=input('Enter the summary text enclosed in quotes\n')
test['count']=input('Enter the number of reviews for this product\n')

test=vect.transform(features(test))
print 'Grocery and Gourmet Foods Review Predictions\n1-Helpful 0-Not helpful\n'

classifier_NB = MultinomialNB()
classifier_NB.fit(X_train, train_targets)
print 'Naive Bayes'
print classifier_NB.predict(test)
    
model_LR = LogisticRegression()
model_LR.fit(X_train, train_targets)
print 'Logistic Regression'
print model_LR.predict(test)

modelTree = DecisionTreeClassifier(max_depth=7)
modelTree.fit(X_train, train_targets)
print 'Decision Tree'
print modelTree.predict(test)

modelGr = GradientBoostingClassifier()
modelGr.fit(X_train, train_targets)
print 'Gradient Boosting'
print np.array(modelGr.predict(test.toarray()))
