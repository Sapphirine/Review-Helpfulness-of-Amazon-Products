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

gzipfilename='reviews_Electronics.json.gz'
df = getDF(gzipfilename)
print 'read'
f=open('elec_rev.txt','w')
c=len(df.index)
dt=[]
for i in xrange(c):
  h2=df['helpful'][i]
  if h2[1]<=9:
    continue
  if h2[0]>=math.ceil(0.7*h2[1]):
    hfactor=1
  else:
    hfactor=0
  reviewText=df['reviewText'][i]
  unixReviewTime=df['unixReviewTime'][i]
  overall=df['overall'][i]
  summary=df['summary'][i]
  dict1={"reviewText":reviewText,"unixReviewTime":unixReviewTime,"overall":overall,"summary":summary,"hfactor":hfactor}
  dt.append(dict1)
print 'completed'
##f.write(json.dumps(dt))
##f.close()


    
