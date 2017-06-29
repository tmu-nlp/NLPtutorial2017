from collections import defaultdict
import numpy as np
import pickle
import random

def SOFTMAX(x):
    return np.exp(x) / np.sum(np.exp(x))

def CREATE_ONEHOT(arg, ids):
    vector = np.zeros(len(ids))
#    print(vector)
#    print(len(ids))
#    print(ids[arg])
    vector[ids[arg]] = 1
    return vector

def FIND_MAX(p):
    y = 0
    for i in range(len(p)):
        if p[i] > p[y]:
            y = i
    return y

def CREATE_FEATURES(x):
    phi = [0 for i in range(len(ids))]
    x = x.lower().split()
    for word in x:
#        print('ids:', ids)
#        print('word:', word)
        phi[ids[word]] += 1
#    print(phi)
    return phi

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

def BACKWORD_NN(net, x, h, p, p_co, err_c):
    d_w_rx = np.zeros((2, len(x_ids)))
    d_w_rh = np.zeros((2, 2))
    d_w_oh = np.zeros((len(y_ids), 2))
    d_b_r = np.zeros(2)
    d_b_o = np.zeros(len(y_ids))
    w_rx, w_rh, w_oh, b_r, b_o = net
    delta_r1 = np.zeros(len(b_r))

    for t in reversed(range(len(x))):
        delta_o = (p_co[t] - p[t])
        err_c += delta_o
        d_w_oh += np.outer(delta_o, h[t])
        d_b_o += delta_o
        delta_r = np.dot(delta_r1, w_rh) + np.dot(delta_o, w_oh)
        delta_r1 = delta_r * (1 - h[t] ** 2)
        d_w_rx += np.outer(delta_r1, x[t])
        d_b_r += delta_r
        if t != 0:
            d_w_rh += np.outer(delta_r1, h[t-1])
    d_z = [d_w_rx, d_w_rh, d_w_oh, d_b_r, d_b_o]

    return d_z, err_c


def UPDATE_WEIGHTS(net, delta, l):
    for w, d in zip(net, delta):
        w += l * d


if __name__ == '__main__':
    epoch = 5
    l = 0.01
#    layer = 1
#    node = 2
    x_ids = defaultdict(lambda: len(x_ids))
    y_ids = defaultdict(lambda: len(y_ids))

    feat_lab = list()
#    net = list()
#    tmp = list()

    with open('../../data/wiki-en-train.norm_pos', 'r') as i_f:
        for line in i_f:
            words = line.lower().split()
            for word in words:
                 x, y = word.split('_')
                 x_ids[x]
                 y_ids[y]

    with open('../../data/wiki-en-train.norm_pos', 'r') as i_f:
        for line in i_f:
            x_list = list()
            y_list = list()
            words = line.lower().split()
            for word in words:
                x, y = word.split('_')
                x_list.append(CREATE_ONEHOT(x, x_ids))
                y_list.append(CREATE_ONEHOT(y, y_ids))
            feat_lab.append((x_list, y_list))

#    net = CREATE_NETWORK(ids, node)
#    for i in range(layer):
#        tmp.append(np.random.uniform(-0.1, 0.1, node))
#    net.append(tmp)

    w_rx = (np.random.rand(2,len(x_ids)) - .5) * .2 * .1
    w_rh = (np.random.rand(2, 2) - .5) * .2 * .1
    w_oh = (np.random.rand(len(y_ids), 2) - .5) * .2 * .1
    b_r = np.zeros(2)
    b_o = np.zeros(len(y_ids))
    net = [w_rx, w_rh, w_oh, b_r, b_o]

    for i in range(epoch):
        random.shuffle(feat_lab)
        err_count = 0
        print('epoch', i+1)
        for x, y_c in feat_lab:
            h, p, y_p = FORWORD_RNN(net, x)
            delta, err_count = BACKWORD_NN(net, x, h, p, y_c, err_count)
            UPDATE_WEIGHTS(net, delta, l)

    with open('network.dump', 'wb') as net_f:
        pickle.dump(net, net_f)

    with open('x_ids', 'wb') as ids_fx:
        pickle.dump(dict(x_ids), ids_fx)
    with open('y_ids', 'wb') as ids_fy:
        pickle.dump(dict(y_ids), ids_fy)
