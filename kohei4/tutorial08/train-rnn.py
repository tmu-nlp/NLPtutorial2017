"""
重み初期値の設定、ラムダ・エポックの設定も神経質な、RNNです。
最終的には、学習時の文へのrandum shuffleが効きました。
チューニングの追い込みまだですが、
この設定で、
kohei$ ../../script/gradepos.pl ../../data/wiki-en-test.pos answer.txt
Accuracy: 75.39% (3440/4563)
"""

# coding: utf-8
from collections import defaultdict
import numpy as np
import math
import sys
import pickle
import random

n_iter = 200
ramda = 0.01

begin = 0
end = 0


x_ids = defaultdict(lambda:len(x_ids))
y_ids = defaultdict(lambda:len(y_ids))
def create_x_ids(x):
    #global ids
    #ids = defaultdict(lambda:len(ids))
    for word in x:
        x_ids[word]

def create_y_ids(x):
    for pos in x:
        y_ids[pos]

def create_one_hot(id,size):
    vec = np.zeros(size)
    vec[id] = 1
    return vec


def init_nn():
    np.random.seed(5)
    w_rx = np.array(np.random.rand(2, l_x_ids)-0.5)/50
    w_rh = np.array(np.random.rand(2,2)-0.5)/50
    b_r = np.array(np.random.rand(2)-0.5)/500
    w_oh = np.array(np.random.rand(l_y_ids, 2)-0.5)/50
    b_o = (np.random.rand(l_y_ids)-0.5)/500
    nn = [w_rx, w_rh, b_r, w_oh, b_o]
    return nn

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
        y[t] = find_max(p[t])
    return h, p, y

def softmax(w, t = 1.0):

    e = np.exp(np.array(w) / t)
    #print(w, e)
    dist = e / np.sum(e)
    return dist

def find_max(x):
    y = 0
    for i in range(len(x)):
        if x[i] > x[y]:
            y = i
    return y

def gradient_rnn(net, x, h, p, y_correct):
    d_nn = init_nn()
    d_r_dush = np.zeros(len(net[2]))
    for t in range(len(x)-1, -1, -1):
        #print(np.shape(y_correct[t]), np.shape(p[t]))
        del_o_dush =  y_correct[t] - p[t]
        #pdf版では最後[t]抜け g_slide版では付き
        d_nn[3] += np.outer(del_o_dush,h[t])
        d_nn[4] += del_o_dush
        d_r = np.dot(d_r_dush,net[1]) + np.dot(del_o_dush,net[3])
        d_r_dush = d_r * (1 - pow(h[t],2))
        d_nn[0] += np.outer(d_r_dush,x[t])
        d_nn[2] += d_r_dush
        if t != 0:
            d_nn[1] += np.outer(d_r_dush,h[t-1])

    return d_nn

def update_weight(net, d_nn,ramda):
    net[0] += ramda * d_nn[0]
    net[1] += ramda * d_nn[1]
    net[2] += ramda * d_nn[2]
    net[3] += ramda * d_nn[3]
    net[4] += ramda * d_nn[4]
    #net = += ramba * d_nn
    #print(net)
    #return net


#train_input = "../../test/05-train-input.txt"
train_input = "../../data/wiki-en-train.norm_pos"

with open(train_input,'r') as f:
    for ij, line in enumerate(f):
        #if ij >= begin and ij <= end:
            x_data=[]
            wordtags = line.strip().split(" ")
            for wordtag in wordtags:
                    word, tag = wordtag.split("_")
                    #print(word.lower(),tag)
                    x_ids[word.lower()]
                    y_ids[tag]

x_ids = {v:k for v, k in x_ids.items()}
y_ids = {v:k for v, k in y_ids.items()}
y_ids_inv = {v:k for k, v in y_ids.items()}

l_x_ids = len(x_ids)
l_y_ids = len(y_ids)
#print(len(y_ids_inv))

with open(train_input,'r') as f:
    feat_lab = []
    for ij, line in enumerate(f):
        #if ij >= begin and ij <= end:

            x_list=[]
            y_list=[]
            wordtags = line.strip().split(" ")
            for wordtag in wordtags:
                word, tag = wordtag.split("_")
                #print(word.lower(),tag)
                x_list.append(create_one_hot(x_ids[word.lower()],l_x_ids))
                y_list.append(create_one_hot(y_ids[tag],l_y_ids))
            feat_lab.append([x_list,y_list])


#print(x_ids)
#print(y_ids)
#print
#print(feat_lab)

net = init_nn()

#for x in net:
#    print(np.shape(x))

for _ in range(n_iter):
    random.shuffle(feat_lab)
    for x_list, y_c_list in feat_lab:
        #print(x_list)
        #print(y_list)
        #print()
        h, p, y = forward_nn(net, x_list)
        #print('h=',h,'\n','p=', p,'\n','y=', y)
        d_nn = gradient_rnn(net, x_list, h, p, y_c_list)

        update_weight(net, d_nn,ramda)



#print(net)
#print(d_nn)
#print(x_ids)

with open('id_files','wb') as ff:
    ids = (x_ids, y_ids, y_ids_inv)
    pickle.dump(ids, ff)
with open('net','wb') as gg:
    pickle.dump(net, gg)
