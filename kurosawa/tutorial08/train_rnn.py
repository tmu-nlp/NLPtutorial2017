import numpy as np
from collections import defaultdict
import pickle
import sys
import random
import math

def softmax(x):
#    print(x)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def find_max(p):
#    print(p)
    y = 0
#    print()
    for i in range(len(p)):
#        print(p[i])
#        print(p[y])
        if p[i] > p[y]:
            y = i
    return y

def make_ids(train_file):
    with open(train_file) as f:
        ids_x = defaultdict(lambda: len(ids_x))
        ids_y = defaultdict(lambda: len(ids_y))
        for line in f:
            words = line.split()
            for word in words:
                x, y = word.split('_')
                x = x.lower()
                ids_x[x]
                ids_y[y]
    return ids_x, ids_y

def create_one_hot(w, ids):
    vec = np.zeros((len(ids),1))
#    print(ids[w])
    if w in ids:
        vec[ids[w]] = 1
    return vec

def make_featlab(train, ids_x, ids_y):
    featlab = []
    for line in train:
        x_vec = []
        y_vec = []
        words = line.split()
        for word in words:
            x, y = word.split('_')
            x = x.lower()
            x_vec.append(create_one_hot(x,ids_x))
            y_vec.append(create_one_hot(y,ids_y))
        featlab.append((x_vec, y_vec))
    return featlab

def forward_rnn(net,x):
    h = []
    p = []
    y = []
    w_rx, w_rh, b_r, w_oh, b_o = net
    for t in range(len(x)):
        if t > 0:
            h.append(np.tanh(np.dot(w_rx, x[t]) + np.dot(w_rh, h[t-1]) + b_r))
        else:
            h.append(np.tanh(np.dot(w_rx, x[t]) + b_r))
#        print(w_oh)
#        print(h[t])
        p.append(softmax(np.dot(w_oh, h[t]) + b_o))
        y.append(find_max(p[t]))
    return h,p,y

def gradient_rnn(net,x,h,p,y):
    w_rx, w_rh, b_r, w_oh, b_o = net
    dw_rx = np.zeros((2,len(ids_x)))
    dw_rh = np.zeros((2,2))
    db_r = np.zeros((2,1))
    dw_oh = np.zeros((len(ids_y),2))
    db_o = np.zeros((len(ids_y),1))
    err_r_ = np.zeros((len(b_r),1))
    for t in reversed(range(len(x))):
#        p_ = create_one_hot(y[t], ids_y)
        err_o_ = y[t] - p[t]
        dw_oh += np.outer(err_o_, h[t])
        db_o += err_o_
#        print(w_oh.shape)
#        print(err_o_.shape)
#        print(np.dot(w_rh,err_r_).shape)
        err_r = np.dot(w_rh, err_r_) + np.dot(w_oh.T, err_o_)
        err_r_ = err_r * (1 - h[t] ** 2)
        dw_rx += np.outer(err_r_, x[t])
        db_r += err_r_
        if t != 0:
            dw_rh += np.outer(h[t-1], err_r_)
        dnet = [dw_rx, dw_rh, db_r, dw_oh, db_o]
    return dnet

def update_weights(net,dnet,lambda_):
    for i in range(len(net)):
        net[i] += lambda_ * dnet[i] 
    return net

def train(epoch, ids_x, ids_y, train_file):
    lambda_ = float(sys.argv[2])
    with open(train_file) as train:
        featlab = make_featlab(train, ids_x, ids_y)
    w_rx = (np.random.rand(2,len(ids_x)) - 0.5) / 5
    w_rh = (np.random.rand(2, 2) - 0.5) / 5
    b_r = np.zeros((2,1))
    w_oh = (np.random.rand(len(ids_y), 2) - 0.5) / 5
    b_o = np.zeros((len(ids_y),1))
    net = [w_rx,w_rh,b_r,w_oh,b_o]
    for i in range(epoch):
        print(i)
        random.shuffle(featlab)
        for x, y in featlab:
            h, p, y_predict = forward_rnn(net,x)
#            print(y)
            dnet = gradient_rnn(net,x,h,p,y)
            net = update_weights(net,dnet,lambda_)
    return net

if __name__ == '__main__':
    train_file = '../../data/wiki-en-train.norm_pos'
    epoch = int(sys.argv[1])
    ids_x, ids_y = make_ids(train_file)
#ids作成済み
    net = train(epoch, ids_x, ids_y, train_file)
    with open('weight_file.byte','wb') as w, open('ids_x_file.byte','wb') as ids_x_data, open('ids_y_file.byte','wb') as ids_y_data:
        pickle.dump(net,w)
        pickle.dump(dict(ids_x),ids_x_data)
        pickle.dump(dict(ids_y),ids_y_data)
