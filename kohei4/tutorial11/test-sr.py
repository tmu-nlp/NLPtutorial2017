"""
Kohei-no-MacBook-Air:tutorial11
kohei$ ../../script/grade-dep.py ../../data/mstparser-en-test.dep sr_output.txt
64.906230% (3011/4639) iter = 10, random_shuffle

"""

import random
from collections import Counter, defaultdict
import math
from nltk.tree import Tree
import sys
import pickle


def ShiftReduceTrain(queue, weights):
    stack = [[0,"ROOT","ROOT"]]
    heads = [-1] * (len(queue) +1 )

    while len(queue) >0 or len(stack) >1:
        #print(stack,queue)
        features = MakeFeatures(stack, queue)
        #print(features)
        s_s = PredictScore(weights['shift'],features)
        s_l = PredictScore(weights['left'],features)
        s_r = PredictScore(weights['right'],features)
        #print(s_s, s_l, s_r)
        if (s_s >= s_l and s_s >= s_r and len(queue)>0) or len(stack) <2:
            stack.append(queue.pop(0))
            #print(stack)
        elif s_l >= s_r:
            heads[stack[-2][0]] = stack[-1][0]
            stack.pop(-2)
        else:
            heads[stack[-1][0]] = stack[-2][0]
            stack.pop(-1)

    return heads


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
    input_file = '../../data/mstparser-en-test.dep'

    deb_n = 1

    data = []
    queue = []
    orig = []
    heads_l =[]

    with open(input_file, 'rt') as ff:
        for jj, line in enumerate(ff):
            #if jj >= deb_n and jj <= deb_n :
                if len(line) == 1:
                    data.append(queue)
                    orig.append(queue[:])
                    queue = []
                else:
                    cols = line.strip().split('\t')
                    #print(cols)
                    queue.append([int(cols[0]),cols[1],cols[3]])

        #print(data)

    with open('sr_weights', 'rb') as ff:
        weights = pickle.load(ff)

    #print(weights)

    for queue in data:
        heads = ShiftReduceTrain(queue, weights )
        heads_l.append(heads)


    with open('sr_output.txt','wt') as ff:
        for ii, orig_q in enumerate(orig):
            for jj, col in enumerate(orig_q):
                print('{}\t{}\t{}\t{}\t{}\t_\t{}\t_'\
                .format(col[0],col[1],col[1],col[2],col[2],heads_l[ii][jj+1]),file=ff)
            print(file=ff)
