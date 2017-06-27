from collections import defaultdict
import numpy as np
import random
import pickle
import math

x_ids = defaultdict(lambda :len(x_ids))
y_ids = defaultdict(lambda :len(y_ids))

def create_onehot(i_d,size):
    vec = np.zeros(size)
    vec[i_d] = 1
    return vec

def forward_rnn(net,x):
    h = [0] * len(x)
    p = [0] * len(x)
    y = [0] * len(x)
    for t in range(len(x)):
        if t > 0:
            h[t] = np.tanh(np.dot(net[0],x[t]) + np.dot(net[1],h[t-1]) + net[2])
        else:
            h[t] = np.tanh(np.dot(net[0],x[t]) + net[2])
        p[t] = softmax(np.dot(net[3],h[t]) + net[4])
        y[t] = findmax(p[t])
    return h,p,y

def softmax(x):
#    sig = sum(math.exp(x)) # mathのexpは行列に使用不可
    sig = np.sum(np.exp(x))
    p = np.exp(x) / sig
#    print(p)
    return p

def findmax(x):
    y = 0
    for i in range(len(x)):
        if x[i] > x[y]:
            y = i
#    print(y)
    return y

def create_net(hid):
    np.random.seed(1)
    w_rx = (np.random.rand(hid,len(x_ids)) - 000.5) * 2 # -1~1
    w_rh = (np.random.rand(hid,hid) - 000.5) * 2
    b_r = np.zeros(hid)
    w_oh = (np.random.rand(len(y_ids),hid) - 000.5) * 2
    b_o = np.zeros(len(y_ids))
    net = [w_rx,w_rh,b_r,w_oh,b_o]
    return net

def gradient_rnn(net,x,h,p,y_,hid): # dd = delta-dash
    dw_rx = np.zeros((hid,len(x_ids)))
    dw_rh = np.zeros((hid,hid))
    db_r = np.zeros(hid)
    dw_oh = np.zeros((len(y_ids),hid))
    db_o = np.zeros(len(y_ids))
    d_net = [dw_rx,dw_rh,db_r,dw_oh,db_o]
    # d_net[0-4] = [dw_rx,dw_rh,db_r,dw_oh,db_o]
    dd_r = np.zeros(len(d_net[2]))
    for t in reversed(range(len(x))):
#        for i in range(len(y_ids)):
#        p_ = create_onehot(y_[t],len(y_ids))
#        print(y_[t])
        dd_o = y_[t] - p[t]
        d_net[3] += np.outer(dd_o,h[t])
        d_net[4] += dd_o
        d_r = np.dot(dd_r,net[1]) + np.dot(dd_o,net[3])
        dd_r = d_r * (1 - h[t] ** 2)
        d_net[0] += np.outer(dd_r,x[t])
        d_net[2] += dd_r
        if t != 0:
            d_net[1] += np.outer(dd_r,h[t-1])
    return d_net


def update_weights(net,d_net,lam):
    net[0] += lam * d_net[0]
    net[1] += lam * d_net[1]
    net[2] += lam * d_net[2]
    net[3] += lam * d_net[3]
    net[4] += lam * d_net[4]

lam = 0.005

if __name__ == '__main__':
    word_data = []
    pos_data = []
    wordpos = []
    with open('../../data/wiki-en-train.norm_pos','r') as i_f:
        for line in i_f:
            word_pos = line.split()
            for i in range(len(word_pos)):
                word,pos = word_pos[i].split('_')
                word_data.append(word.lower())
                pos_data.append(pos)
                wordpos.append([word.lower(),pos])
                x_ids[word.lower()]
                y_ids[pos]
#            print(word)

    with open('../../data/wiki-en-train.norm_pos','r') as i_f:
        feat_lab = []
        for line in i_f:
            word_list = []
            pos_list = []
            word_pos = line.split()
            for i in range(len(word_pos)):
                word,pos = word_pos[i].split('_')
                word_list.append(create_onehot(x_ids[word.lower()],len(x_ids)))
                pos_list.append(create_onehot(y_ids[pos],len(y_ids)))
            feat_lab.append([word_list,pos_list])
#        print(feat_lab)
    epoch = int(input('epoch :'))
    hide = int(input('hidden layer :'))

    rnn_net = create_net(hide)
#    print(np.shape(rnn_net[]))
    for i in range(epoch):
        random.shuffle(feat_lab)
        for x,y_ in feat_lab:
#            print(np.shape(x))
            h,p,y = forward_rnn(rnn_net,x)
#            print(y_)
            delta_net = gradient_rnn(rnn_net,x,h,p,y_,hide)
            update_weights(rnn_net,delta_net,lam)
        print('\nepoch:{}\tcomplete!\n'.format(i+1))

    with open('weight_file','wb') as w_f:
        pickle.dump(rnn_net,w_f)
    with open('id_file','wb') as id_f:
        ids = (dict(x_ids),dict(y_ids),word_data,pos_data)
        pickle.dump(ids,id_f)
