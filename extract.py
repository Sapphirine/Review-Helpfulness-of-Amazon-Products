import sys
import json
import pandas as pd
import math
import gzip

def parse(path):
  g = gzip.open(path, 'rb')
  
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')


gzipfilename=sys.argv[1]
df = getDF(gzipfilename)

prodfilename=sys.argv[2]
prodf = getDF(prodfilename)



f=open('grocery.txt','w')
c=len(df.index)

time_dict={}
count_dict={}

for i in xrange(c):
    pid=df['asin'][i]
    tim=df['unixReviewTime'][i]
    if pid in time_dict:
        count_dict[pid]=count_dict[pid]+1
        if tim<time_dict[pid]:
            time_dict[pid]=tim
    else:
        count_dict[pid]=1
        time_dict[pid]=tim

cat_dict={}
for i in xrange(len(prodf)):
    pid=prodf['asin'][i]
    if pid not in cat_dict:
        cat_dict[pid]=prodf['categories'][i]

dt=[]
for i in xrange(c):
    pid=df['asin'][i]
    h2=df['helpful'][i]
    if h2[1]<=9:
        continue
    if h2[0]>=math.ceil(0.7*h2[1]):
        hfactor=1
    else:
        hfactor=0
    reviewText=df['reviewText'][i]
    unixReviewTime=df['unixReviewTime'][i]-time_dict[pid]
    overall=df['overall'][i]
    summary=df['summary'][i]
    dict1={"reviewText":reviewText,"unixReviewTime":unixReviewTime,"overall":overall,"summary":summary,"count":count_dict[pid],"categories":cat_dict[pid],"hfactor":hfactor}
    dt.append(dict1)
f.write(json.dumps(dt))
f.close()


    
