
# coding: utf-8

# In[1]:

import gzip
from collections import defaultdict

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

### Rating baseline: compute averages for each user, or return the global average if we've never seen the user before

allRatings = []
counter_training = 0
userRatings = defaultdict(list)
data = list(readGz("/home/ygao/Downloads/pycharm-4.5.4/assignment1/train.json.gz"))
data_training = data[:100000]
data_validation = data[900000:]


# In[2]:

allHelpful = []
userHelpful = defaultdict(list)

for l in data_training:
    user,item,helpful = l['reviewerID'], l['itemID'], l['helpful']
    allHelpful.append(helpful)

    


# In[3]:

# question 1 algorithm 1
sumHelpful = sum(x['nHelpful'] for x in allHelpful) * 1.0


# In[4]:

sumTotal = sum(x['outOf'] for x in allHelpful)


# In[5]:

sumTotal, sumHelpful


# In[6]:

averageHelpful = sumHelpful / sumTotal


# In[7]:

averageHelpful


# In[12]:

# question 1 algorithm 2
total = 0
for l in data_training:
    nHelpful, outOf = l['helpful']['nHelpful'], l['helpful']['outOf']
    if outOf != 0:
        total += nHelpful * 1.0 / outOf
averageHelpful = total / len(data_training)


# In[13]:

averageHelpful


# In[14]:

# question 2
AE = 0
for l in data_validation:
    helpful = l['helpful']
    ratio = 0.0
    predicted = averageHelpful * helpful['outOf']
    AE += abs(predicted - helpful['nHelpful'])
MAE = AE / len(data_validation)


# In[15]:

MAE


# In[32]:

# question 3
X = []
Y = []
for l in data_training:
    nHelpful,outOf,rating = l['helpful']['nHelpful'], l['helpful']['outOf'], l['rating']
    text = l['reviewText']
    numberOfWord = len(text.split())
    ratio = 0
    if outOf == 0:
        continue
    if outOf != 0:
        ratio = nHelpful * 1.0 / outOf
    Y.append(ratio)
    feat = [1]
    feat.append(numberOfWord)
    feat.append(rating)
    X.append(feat)


# In[33]:

import numpy as np
np.linalg.lstsq(X ,Y)


# In[34]:

alpha, theta1, theta2 = np.linalg.lstsq(X, Y)[0]


# In[35]:

alpha


# In[36]:

theta1


# In[ ]:




# In[37]:

totalError = 0
totalRatio = 0
for l in data_validation:
    nHelpful,outOf,rating = l['helpful']['nHelpful'], l['helpful']['outOf'], l['rating']
    text = l['reviewText']
    numberOfWord = len(text.split())
    predict = alpha + theta1 * numberOfWord + theta2 * rating
    totalError += abs(predict * outOf - nHelpful)
    ratio = 0
    if outOf != 0:
        ratio = nHelpful * 1.0 / outOf
    totalRatio += abs(ratio - predict)
print totalError
print totalRatio


# In[38]:

totalError / len(data_validation)


# In[39]:

totalRatio / len(data_validation)


# In[40]:

# question 4


# In[41]:

words = defaultdict(float)
ratings = defaultdict(float)
data_prediction = list(readGz("/home/ygao/Downloads/pycharm-4.5.4/assignment1/helpful.json.gz"))
for l in data_prediction:
    user, item, rating, text = l['reviewerID'], l['itemID'], l['rating'], l['reviewText']
    numberOfWord = len(text.split())
    words[user, item] = numberOfWord
    ratings[user, item] = rating


# In[42]:

len(words)


# In[43]:

len(ratings)


# In[ ]:




# In[44]:

predictions = open("/home/ygao/Downloads/pycharm-4.5.4/assignment1/predictions_Helpful.txt", 'w')
for l in open("/home/ygao/Downloads/pycharm-4.5.4/assignment1/pairs_Helpful.txt"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,i,outOf = l.strip().split('-')
    outOf = int(outOf)
    predictions.write(u + '-' + i + '-' + str(outOf) + ',' + str(outOf*(alpha + words[u, i] * theta1 + ratings[u, i] * theta2)) + '\n')      
predictions.close()

