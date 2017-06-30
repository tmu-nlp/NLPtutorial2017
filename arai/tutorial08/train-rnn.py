import numpy as np
from collections import defaultdict
import sys
import pickle
import random

def create_one_hot(iddayo,size):
    vec = np.zeros(size)
    vec[iddayo] = 1
    return vec

def softmax(array):
    e = np.exp(array - np.max(x))
    return e / e.sum()

def forward_rnn(net, x):
    t = 0
    h = [] 
    p = []
    y_predict = []
    for t in range(0,len(x)):
        if t >0:
            h.append(np.tanh(w_rx.dot(x[t]) + w_rh.dot(h[t-1]) + b_r))
        else:
            h.append(np.tanh(w_rx.dot(x[t]) + b_r))
        p.append(softmax(w_oh.dot(h[t]) + b_o))
        y_predict.append(np.argmax(p[t]))
    return h, p, y_predict

def gradient_rnn(net,x,h,p,y_):
            #print(x[t])
    net[0] = w_rx
    net[1] = w_rh 
    net[2] = b_r
    net[3] = w_oh
    net[4] = b_o
    d_weight_rx = np.array((np.random.rand(2, len(x_ids))-0.5)/5) 
    d_weight_rh = np.array((np.random.rand(2,2)-0.5) / 5)
    d_bias_r = (np.random.rand(2) - 0.5) / 5
    d_weight_oh = np.array((np.random.rand(len(y_ids),2)-0.5)/5)
    d_bias_o = (np.random.rand(len(y_ids)) - 0.5) /5
    delta_r_ = np.zeros(len(b_r))
    for time in range(len(x) -1, -1, -1):
        delta_o_ = p[time] - y_[time]
        d_weight_oh += np.outer(delta_o_, h[time])
        d_bias_o += delta_o_
        delta_r = w_rh.dot(delta_r_) + delta_o_.dot(w_oh)
        delta_r_ = delta_r * (1 - h[time] ** 2)
        d_weight_rx += np.outer(delta_r_, x[time])
        d_bias_r += delta_r_

        if time != 0:
            d_weight_rh += np.outer(delta_r_, h[time - 1])
    deltas = [d_weight_rx, d_weight_rh, d_bias_r, d_weight_oh, d_bias_o]
    return deltas

    #update_weights(net, d_weight_rx, d_weight_rh, d_bias_r, d_weight_oh, d_bias_o, lambda1)

def update_weights(net, delta, lambda1):
    net[0] -= lambda1 * delta[0]
    net[1] -= lambda1 * delta[1]
    net[2] -= lambda1 * delta[2]
    net[3] -= lambda1 * delta[3]
    net[4] -= lambda1 * delta[4]



if __name__ == '__main__':
    x_ids = defaultdict(lambda: len(x_ids))
    y_ids = defaultdict(lambda: len(y_ids))
    x_list = []
    y_list = []
    l = 1
    feat_lab = []
    lambda1 = 0.005
    #all_x_counts = {}
    #all_y_counts = {}
    with open('../../data/wiki-en-train.norm_pos') as text:
        for line in text:
            #x_list = []
            #y_list = []
            words= line.strip().split()
            for word in words:
                x, y = word.split('_')
                x_ids[x]
                y_ids[y]
                '''
                if x in all_x_counts: 
                    all_x_counts[x] += 1
                else:
                    all_x_counts[x] = 1
                if y in all_y_counts:
                    all_y_counts[y] += 1
                else:
                    all_y_counts[y] = 1
                    '''
                #print(x,y)
        #print(len(all_x_counts),len(all_y_counts))
                #print(y_ids)
   
    with open('../../data/wiki-en-train.norm_pos') as text:
        #feat_lab =[]
        for line in text:
            #x_list = []
            #y_list = []
            words= line.strip().split()
            for word in words:
                x, y = word.split('_')
                x_list.append(create_one_hot(x_ids[x],len(x_ids)))
                y_list.append(create_one_hot(y_ids[y],len(y_ids)))
            feat_lab.append([x_list, y_list])
        #print(x_list)
    w_rx = np.array((np.random.rand(2, len(x_ids))-0.5)/5) 
    w_rh = np.array((np.random.rand(2,2)-0.5) / 5)
    b_r = (np.random.rand(2) - 0.5) / 5
    w_oh = np.array((np.random.rand(len(y_ids),2)-0.5)/5)
    b_o = (np.random.rand(len(y_ids)) - 0.5) /5
    net = [w_rx, w_rh, b_r, w_oh, b_o]

    #print(x_list[-1])
    for i in range(l):
        random.shuffle(feat_lab)
        for x,y_ in feat_lab:
            h,p,y_predict = forward_rnn(net,x)
            delta = gradient_rnn(net,x,h,p,y_)
            update_weights(net, delta,lambda1)
            print(y_)
#print(h, p, y_predict)

with open ('weight_file.dump', 'wb') as w_f :
    pickle.dump(net, w_f)
with open ('x_ids_file.dump', 'wb') as xids_f:
    pickle.dump(dict(x_ids), xids_f)
with open('y_ids_file.dump', 'wb') as yids_f:
    pickle.dump(dict(y_ids), yids_f)
  

            

