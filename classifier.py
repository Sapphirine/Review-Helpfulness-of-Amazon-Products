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


train,test=train_test_split(data, train_size=0.8)

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

print 'Grocery and Gourmet Foods reviews\n'

classifier_NB = MultinomialNB()
classifier_NB.fit(X_train, train_targets)
predictions_NB = classifier_NB.predict(X_test)
print 'Naive Bayes'
print 'Accuracy score: '+str(accuracy_score(test_targets,predictions_NB))
print 'Precision score: '+str(precision_score(test_targets,predictions_NB))
    
model_LR = LogisticRegression()
model_LR.fit(X_train, train_targets)
predictions_LR = model_LR.predict(X_test)
print 'Logistic Regression'
print 'Accuracy score: '+str(accuracy_score(test_targets,predictions_LR))
print 'Precision score: '+str(precision_score(test_targets,predictions_LR))

modelTree = DecisionTreeClassifier(max_depth=7)
modelTree.fit(X_train, train_targets)
predictions_Tree = modelTree.predict(X_test)
print 'Decision Tree'
print 'Accuracy score: '+str(accuracy_score(test_targets,predictions_Tree))
print 'Precision score: '+str(precision_score(test_targets,predictions_Tree))


modelGr = GradientBoostingClassifier()
modelGr.fit(X_train, train_targets)
predictions_Gr = np.array(modelGr.predict(X_test.toarray()))
print 'Gradient Boosting'
print 'Accuracy score: '+str(accuracy_score(test_targets,predictions_Gr))
print 'Precision score: '+str(precision_score(test_targets,predictions_Gr))
