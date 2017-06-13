"""
PKohei-no-MacBook-Air:tutorial06 kohei$
../../script/grade-prediction.py ../../data/titles-en-test.labeled my_answer
Accuracy = 92.029756%
n_iter = 10
eta = 1
margin = 0.1
c = 0.0001

n_iter = 10
eta = 1
margin = 50
c = 0.0001


"""

import sys
import math
from collections import defaultdict

n_iter = 10
eta = 1
margin = 50
c = 0.0001
#train_input = "../../data/03-train-input.txt"
train_input = "../../data/titles-en-train.labeled"
model_file = "svm_model.txt"
input_file = "../../data/titles-en-test.word"
w = dict()
data = []

def create_features(x):
    phi = defaultdict(lambda: 0)

#    words = x.split()
    for word in x:
        phi["UNI:" + word] += 1

    return phi

def w_phi(w,phi):
    score=0
    for name in phi.keys():
        if (name in w) == False:
            w[name] = 0

        score += phi[name]*w[name]

        #print(w.items())
    return score


def update_w(w,phi,y,c):
    for name, value in w.items():
        if abs(value) <c:
            w[name] = 0
        else:
            w[name] -= math.copysign(1,value)*c

    for name, value in phi.items():
        w[name] += eta*value*y
        #print(name, value)








with open(train_input,'r') as f:
    for line in f:
        words = line.split()
        #print(words[1:])
        data.append([int(words[0]),words[1:]])

#print(data)
for _ in range(n_iter-1):
    for y, x in data:
        phi = create_features(x)
        #print(phi.items())
        n_w_phi = w_phi(w,phi)
        #print(y_new)
        val = n_w_phi * y
        if val <= margin:
            update_w(w,phi,y,c)

for key,value in w.items():
    #print(key,value)
    with open('svm_model.txt','a') as fw:
        fw.write("{}\t{}\n".format(key, value))

def predict_one(w,phi):
    score=0
    for name in phi.keys():
        if (name in w) == False:
            w[name] = 0
        #print(type(phi[name]), type(w[name])
        score += phi[name]*w[name]

        #print(w.items())

    if score >= 0:
        return 1
    else:
        return -1

def predict_all(model_file, input_file):
    with open(model_file, 'r') as f:
        for line in f:
            weights = line.rstrip('\n').split('\t')
            w[weights[0]] = float(weights[1])
        #print(w.items())

    with open(input_file, 'r') as ff:
        for line in ff:
            words = line.split()
            phi = create_features(words)
            y_predicted = predict_one(w,phi)
            print(y_predicted)


predict_all(model_file, input_file)
