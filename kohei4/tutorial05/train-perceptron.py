"""
Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications (updates) in each epoch.
"""

import sys
import math
from collections import defaultdict

n_iter = 10
eta = 1
#train_input = "../../data/03-train-input.txt"
train_input = "../../data/titles-en-train.labeled"
w = dict()
data = []

def create_features(x):
    phi = defaultdict(lambda: 0)

#    words = x.split()
    for word in x:
        phi["UNI:" + word] += 1

    return phi

def predict_one(w,phi):
    score=0
    for name in phi.keys():
        if (name in w) == False:
            w[name] = 0

        score += phi[name]*w[name]

        #print(w.items())

    if score >= 0:
        return 1
    else:
        return -1

def update_w(w,phi,y):
    for name in phi.keys():
        w[name] += eta*phi[name]*y


with open(train_input,'r') as f:
    for line in f:
        words = line.split()
        #print(words[1:])
        data.append([int(words[0]),words[1:]])

#print(data)
for _ in range(n_iter):
    for y, x in data:
        phi = create_features(x)
        #print(phi.items())
        y_new = predict_one(w,phi)
        #print(y_new)
        if y_new != y:
            update_w(w,phi,y)

for key,value in w.items():
    #print(key,value)
    print("{}\t{}".format(key, value))
