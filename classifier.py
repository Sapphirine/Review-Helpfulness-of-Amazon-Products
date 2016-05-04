import time
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
import logging
import math
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


logging.captureWarnings(True)
with open('aiv.txt') as datafile:
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
##    categories=d['categories']
##    
##    for c in categories:
##        data[c[0]]=1.0
##    data['text_length']=len(termsText)
##    data['summary_length']=len(termsSummary)
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


classifier_NB = MultinomialNB()
classifier_NB.fit(X_train, train_targets)
predictions_NB = classifier_NB.predict(X_test)

print accuracy_score(test_targets,predictions_NB)

##testing={'overall':5.0
##         ,'summary':'KISS HER, KILL HER, BE HER'
##         ,'reviewText':'First off, this is part of a series. If you haven\'t seen the previous films (or read the books) you will be hopelessly lost. There is no flashback to bring you up to date.This film picks up where the last one leaves off. Katniss (Jennifer Lawrence) is recruited to be the face of the rebellion. She is reluctant until it becomes personal. Happy content people don\'t pick fights. The film consists of a propaganda war between Peeta (Josh Hutcherson) and the government and Katniss and the rebels/terrorists/anarchists/freedom fighters. It finishes on an up note...sort of.The film delves into the problems of a rebellion and the need for a war of words to convince hearts and minds. In many ways I like this film more than the previous ones, especially the second one which was just a filler film to get us to the rebellion. I think of the second film as "Star Wars Episode 2" but less annoying. My apologies to the president of the Jar Jar Binks fan club.In this film, I now see why Jennifer Lawrence was needed for the series as she transforms from reality TV star to a person who runs a gauntlet of emotions: fear, despair, terror, panic, hopelessness, and pretending she can\'t act. Clearly she nailed it as only she can.The film had a number of memorable lines:"My sister gets to keep her cat." was great. It demonstrates that the rebellion is about her personal life and that it takes priority over any cause...for better or worse."Best dressed rebel in history." Shows some mindless priorities."Prepare to pay the ultimate price."And with sadly Philip Seymour Hoffman says, "Anyone can be replaced."Katniss takes a bow to a gun fight. It makes for good fiction, but seriously?If you have seen the other two, you are going to watch this one too.Guide: No bad language, sex, or nudity.'
##         ,'unixReviewTime':math.ceil(time.time())}
##t=features(testing)
##test=vect.transform(t)
    
model_LR = LogisticRegression()
model_LR.fit(X_train, train_targets)
predictions_LR = model_LR.predict(X_test)

print accuracy_score(test_targets,predictions_LR)
