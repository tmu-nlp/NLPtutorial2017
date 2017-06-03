"""

Kohei-no-MacBook-Air:tutorial05 kohei$
 ../../script/grade-prediction.py ../../data/titles-en-test.labeled my_answer
Accuracy = 93.446688%
"""

import sys
import math
from collections import defaultdict

n_iter = 10
eta = 1
#train_input = "../../data/03-train-input.txt"
model_file = "per_model.txt"
input_file = "../../data/titles-en-test.word"
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

def predict_all(model_file, input_file):
    with open(model_file, 'r') as f:
        for line in f:
            weights = line.rstrip('\n').split('\t')
            w[weights[0]] = int(weights[1])
        #print(w.items())

    with open(input_file, 'r') as ff:
        for line in ff:
            words = line.split()
            phi = create_features(words)
            y_predicted = predict_one(w,phi)
            print(y_predicted)


predict_all(model_file, input_file)
