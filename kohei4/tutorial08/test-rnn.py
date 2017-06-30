"""
未知語の処理はこれで良いのか？
"""
# coding: utf-8
from collections import defaultdict
import numpy as np
import math
import sys
import pickle

answerfile = 'answer.txt'
with open(answerfile,'w') as gg:
    print("Answer", file = gg)


begin = 0
end = 3

def forward_nn(net, x):
    h = [0 for i in range(len(x))]
    p = [0 for i in range(len(x))]
    y = [0 for i in range(len(x))]
    for t in range(len(x)):
        if t > 0:
            h[t] = np.tanh(np.dot(net[0],x[t]) + np.dot(net[1],h[t-1]) + net[2])
        else:
            h[t] = np.tanh(np.dot(net[0],x[t]) + net[2])
        p[t] = softmax(np.dot(net[3],h[t]) + net[4] )
        #print(p[t])
        y[t] = find_max(p[t])
        #print(y[t])
    return h, p, y

def softmax(w, t = 1.0):
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    return dist

def find_max(x):
    y = 0
    for i in range(len(x)):
        if x[i] > x[y]:
            y = i
    return y

def create_one_hot(id,size):
    vec = np.zeros(size)
    vec[id] = 1
    return vec



##### main ###########

with open('net','rb') as ff:
     net = pickle.load(ff)
     #print(net)
with open('id_files','rb') as gg:
    x_ids, y_ids, y_ids_inv = pickle.load(gg)
    x_ids_size = len(x_ids)
    #print(len(x_ids))
    #print(x_ids_size)
    #print(y_ids)


#test_input = "../../test/05-test-input.txt"
test_input = "../../data/wiki-en-test.norm"



with open(test_input,'r') as f:
    feat = []
    for jj, line in enumerate(f):
        #if jj >= begin and jj <= end:
            x_list=[]
            unknown=[]
            words = line.strip().split(" ")
            for ii, word in enumerate(words):
                if word.lower() in x_ids:
                    #print(x_ids[word.lower()])
                    x_list.append(create_one_hot(x_ids[word.lower()],len(x_ids)))
                else:
                    #x_list.append(np.zeros(len(x_ids)))
                    x_list.append(np.ones(len(x_ids))/len(x_ids))
                    unknown.append(ii)
                    print('!',word.lower())

            #print(x_list)
            h, p, y_list = forward_nn(net, x_list)
            #print(y_list)
            y_predict = [ y_ids_inv[x] for x in y_list]
            #if len(unknown) > 0:
                #print('unknown line',jj)
                #for i, ii in enumerate(unknown):
                    #y_predict.insert(ii+i,'UK')


            with open(answerfile, 'a') as hh:
                print(*y_predict, file = hh)
