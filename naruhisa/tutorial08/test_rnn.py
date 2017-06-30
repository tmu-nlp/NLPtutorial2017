from collections import defaultdict
import numpy as np
import pickle
import random

def FORWORD_RNN(network, x):
    h = list()
    p = list()
    y = list()
    w_rx, w_rh, w_oh, b_r, b_o = network
    for t in range(len(x)):
        if t > 0:
            h.append(np.tanh(np.dot(w_rx, x[t]) + np.dot(w_rh, h[t - 1] + b_r)))
        else:
            h.append(np.tanh(np.dot(w_rx, x[t]) + b_r))
        p.append(SOFTMAX(np.dot(w_oh, h[t]) + b_o))
        y.append(FIND_MAX(p[t]))
    return h, p, y

def SOFTMAX(x):
    return np.exp(x) / np.sum(np.exp(x))

def id_to_pos(num):
    return rev_yids[num]

def CREATE_ONEHOT(arg, ids):
    vector = np.zeros(len(ids))
#    print(vector)
#    print(len(ids))
#    print(ids[arg])
    if arg in ids:
        vector[ids[arg]] = 1
    return vector

def FIND_MAX(p):
    y = 0
    for i in range(len(p)):
        if p[i] > p[y]:
            y = i
    return y

if __name__ == '__main__':
    with open('network.dump', 'rb') as net_f:
        net = pickle.load(net_f)
    with open('x_ids', 'rb') as ids_f1:
        x_ids = pickle.load(ids_f1)
    with open('y_ids', 'rb') as ids_f2:
        y_ids = pickle.load(ids_f2)
    rev_yids = dict()
    for key, value in y_ids.items():
        rev_yids[value] = key
    with open('my_answer.rnn', 'w') as ans_f, open('../../data/wiki-en-test.norm', 'r') as t_f:
        for line in t_f:
            x_list = list()
            for x in line.split():
                x_list.append(CREATE_ONEHOT(x, x_ids))
            h, p, y_list = FORWORD_RNN(net, x_list)
            y_predicts = map(id_to_pos, y_list)
            ans_f.write (' '.join(y_predicts))
            ans_f.write('\n')
