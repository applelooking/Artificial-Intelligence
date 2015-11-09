
# coding: utf-8

# In[5]:

import gzip
from collections import defaultdict

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

### Rating baseline: compute averages for each user, or return the global average if we've never seen the user before

allRatings = []
counter_training = 0
userRatings = defaultdict(list)


# In[6]:

data = list(readGz("/home/ygao/Downloads/pycharm-4.5.4/assignment1/train.json.gz"))


# In[7]:

len(data)


# In[10]:

data_training = data[:100000]


# In[12]:

len(data_training)


# In[16]:

data_validation = data[900000:]


# In[18]:

len(data_validation)


# In[19]:

for l in data_training:
    user, item = l['reviewerID'], l['itemID']
    allRatings.append(l['rating'])
    userRatings[user].append(l['rating'])


# In[20]:

len(allRatings)


# In[21]:

len(userRatings)


# In[22]:

# question 5 
globalAverage = sum(allRatings) / len(allRatings)


# In[23]:

globalAverage


# In[40]:

totalSquareError = 0


# In[43]:

validationRatings = []
for l in data_validation:
    rating = l['rating']
    totalSquareError += (globalAverage - rating)**2


# In[48]:

import math
RMSE = math.sqrt(totalSquareError / len(data_validation))
MSE = totalSquareError / len(data_validation)


# In[50]:

RMSE


# In[51]:

MSE


# In[35]:

userAverage = {}
for u in userRatings:
    userAverage[u] = sum(userRatings[u]) / len(userRatings[u])


# In[143]:

# question 6
alpha = globalAverage
betaUser = defaultdict(float)
betaItem = defaultdict(float)

userRatings = defaultdict(list)
itemRatings = defaultdict(list)
userAndItemRatings = defaultdict(float)
userList = []
itemList = []


# In[144]:

for l in data_training:
    user,item,rating = l['reviewerID'], l['itemID'], l['rating']
    userRatings[user].append(item)
    itemRatings[item].append(user)
    userAndItemRatings[user, item] = rating
    userList.append(user)
    itemList.append(item)
userSet = set(userList)
itemSet = set(itemList)


# In[145]:

def updateAlpha():
    global alpha
    total = 0
    for user, item in userAndItemRatings:
        total += userAndItemRatings[user, item]
        total -= betaUser[user]
        total -= betaItem[item]
    alpha = total / len(userAndItemRatings)
    return alpha


# In[146]:

def updateBetaUser(user, lamb):
    global alpha
    total = 0
    for item in userRatings[user]:
        total += userAndItemRatings[user, item]
        total -= (alpha + betaItem[item])
    betaUser[user] = total / (lamb + len(userRatings[user]))
    


# In[147]:

def updateBetaItem(item, lamb):
    global alpha
    total = 0
    for user in itemRatings[item]:
        total += userAndItemRatings[user, item]
        total -= (alpha + betaUser[user])
    betaItem[item] = total / (lamb + len(itemRatings[item]))


# In[148]:

for i in range(200):
    global alpha 
    alpha = updateAlpha()
    totalUser = 0
    totalItem = 0
    for user in userSet:
        updateBetaUser(user, 4.0)
        if totalUser > betaUser[user]:
            totalUser += betaUser[user]
            totalUser = betaUser[user]
    for item in itemSet:
        updateBetaItem(item, 4.0)
        #totalItem = b
        totalItem += betaItem[item]
    #print i, alpha, totalUser, totalItem
    


# In[149]:

totalSE = 0
for l in data_validation:
    user, item, rating = l['reviewerID'], l['itemID'], l['rating']
    predict = alpha + betaUser[user] + betaItem[item]
    totalSE += (predict - rating)**2
totalSE /= len(data_validation)


# In[150]:

totalSE


# In[120]:

# question 7
import operator
maxUser = max(betaUser.iteritems(), key=operator.itemgetter(1))


# In[121]:

maxUser


# In[123]:

minUser = min(betaUser.iteritems(), key=operator.itemgetter(1))


# In[124]:

minUser


# In[125]:

maxItem = max(betaItem.iteritems(), key=operator.itemgetter(1))
maxItem


# In[126]:

minItem = min(betaItem.iteritems(), key=operator.itemgetter(1))
minItem


# In[137]:

# question 8

userRatings = defaultdict(list)
itemRatings = defaultdict(list)
userAndItemRatings = defaultdict(float)
userList = []
itemList = []
for l in data_training:
    user,item,rating = l['reviewerID'], l['itemID'], l['rating']
    userRatings[user].append(item)
    itemRatings[item].append(user)
    userAndItemRatings[user, item] = rating
    userList.append(user)
    itemList.append(item)
userSet = set(userList)
itemSet = set(itemList)
lamMSE = defaultdict(float)


# In[142]:

for i in range(20):
    lamb = i
    alpha = globalAverage
    betaUser = defaultdict(float)
    betaItem = defaultdict(float)
    for j in range(200):
        alpha = updateAlpha()
        totalUser = 0
        totalItem = 0
        for user in userSet:
            updateBetaUser(user, i)
        for item in itemSet:
            updateBetaItem(item, i)
    totalSE = 0
    for l in data_validation:
        user, item, rating = l['reviewerID'], l['itemID'], l['rating']
        predict = alpha + betaUser[user] + betaItem[item]
        totalSE += (predict - rating)**2
    totalSE /= len(data_validation)     
    lamMSE[i] = totalSE
    print i, totalSE
    #print i, alpha, totalUser, totalItem
    


# In[140]:

lamMSE


# In[155]:

predictions = open("/home/ygao/Downloads/pycharm-4.5.4/assignment1/predictions_Rating.txt", 'w')
for l in open("/home/ygao/Downloads/pycharm-4.5.4/assignment1/pairs_Rating.txt"):
    if l.startswith("userID"):
        #header
        predictions.write(l)
        continue
    u,i = l.strip().split('-')
    predictions.write(u + '-' + i + ',' + str(alpha + betaUser[u] + betaItem[i]) + '\n')
predictions.close()


# In[ ]:



