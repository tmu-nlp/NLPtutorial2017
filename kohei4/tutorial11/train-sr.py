import random
from collections import Counter, defaultdict
import math
from nltk.tree import Tree
import sys
import pickle


def ShiftReduceTrain(queue, heads, weights):
    stack = [[0,"ROOT","ROOT"]]
    unproc = []
    for i in range(len(heads)):
        unproc.append(heads.count(i))

    #print(upproc)
    #for ii in range(1):
    while len(queue)>0 or len(stack) >1:
        features = MakeFeatures(stack, queue)
        #print(features)
        s_s = PredictScore(weights['shift'],features)
        s_l = PredictScore(weights['left'],features)
        s_r = PredictScore(weights['right'],features)
        #print(s_s, s_l, s_r)
        if (s_s >= s_l and s_s >= s_r and len(queue)>0) or len(stack) <2:
            predict = "shift"
        elif s_l >= s_r:
            predict = "left"
        else:
            predict = "right"

        if len(stack) < 2 :
            correct = "shift"

        elif heads[stack[-1][0]] is stack[-2][0] and unproc[stack[-1][0]] is 0:
            correct = "right"
        elif heads[stack[-2][0]] is stack[-1][0] and unproc[stack[-2][0]] is 0:
            correct = "left"
        else:
            correct = 'shift'

        if predict is not correct:
            weights = UpdateWeights(weights, features, predict,correct)

        if correct is "shift":
            stack.append(queue.pop(0))

        elif correct is "left":
            unproc[stack[-1][0]] -= 1
            stack.pop(-2)

        elif correct is "right":
            unproc[stack[-2][0]] -= 1
            stack.pop(-1)


def UpdateWeights(weights, features, predict,correct):
    for key in features.keys():
        weights[predict][key] -= features[key]
        weights[correct][key] += features[key]
    return weights

def PredictScore(weight, features):
    score = 0
    for key in features.keys():
        score += weight[key] * features[key]

    return score



def MakeFeatures(stack, queue):
    features = defaultdict(int)
    if len(queue) > 0 and len(stack) >0:
        features["W-1" + stack[-1][1] + ",W0" + queue[0][1]] += 1
        features["W-1" + stack[-1][1] + ",P0" + queue[0][2]] += 1
        features["P-1" + stack[-1][2] + ",W0" + queue[0][1]] += 1
        features["P-1" + stack[-1][2] + ",P0" + queue[0][2]] += 1
    if len(stack) > 1:
        features["W-2" + stack[-2][1] + ",W-1" + stack[-1][1]] += 1
        features["W-2" + stack[-2][1] + ",P-1" + stack[-1][2]] += 1
        features["P-2" + stack[-2][2] + ",W-1" + stack[-1][1]] += 1
        features["P-2" + stack[-2][2] + ",P-1" + stack[-1][2]] += 1
    return features
        #print("W-1" + stack[-1][1] + ",W0" + queue[0][1])

if __name__ == "__main__":

    #g_file = '../../test/08-grammar.txt'
    input_file = '../../data/mstparser-en-train.dep'

    deb_n = 0
    iter_n = 1

    data = []
    queue = []
    heads = [-1]

    with open(input_file, 'rt') as ff:
        for line in ff:
            if len(line) == 1:
                data.append([queue,heads])
                queue = []
                heads = [-1]
            else:
                cols = line.strip().split('\t')
                #print(cols)
                queue.append([int(cols[0]),cols[1],cols[3]])
                heads.append(int(cols[6]))

        #print(data)

    weights = dict()
    weights['shift'] = defaultdict(int)
    weights['left'] = defaultdict(int)
    weights['right'] = defaultdict(int)



    for ii in range(iter_n):
        random.shuffle(data)
        for jj, (queue, heads) in enumerate(data):
            #if jj >= deb_n and jj <= deb_n :
                ShiftReduceTrain(queue, heads, weights )

    #print(weights)

    with open('sr_weights', 'wb') as ff:
        pickle.dump(weights,ff)
