from collections import defaultdict
import numpy as np
import pickle
import random

def CREATE_NETWORK(x, node):
    net = list()
    tmp = list()
    tmp_w = np.random.uniform(-0.1, 0.1, len(x))
    tmp_b = [0, 0]
    tmp = [tmp_w, tmp_b]
    net.append(tmp)
    tmp_w = np.random.uniform(-0.1, 0.1, 2)
    tmp_b = [0]
    tmp = [tmp_w, tmp_b]
    net.append(tmp)
    return net



def CREATE_FEATURES(x):
    phi = [0 for i in range(len(ids))]
    x = x.lower().split()
    for word in x:
#        print('ids:', ids)
#        print('word:', word)
        phi[ids[word]] += 1
#    print(phi)
    return phi

def FORWORD_NN(network, phi_0):
    phi = [0 for i in range(len(network) + 1)]
    phi[0] = phi_0
#    print(network)
#    print(phi)
    for i in range(1, len(network) + 1):
        w, b = network[i - 1]
        phi[i] = np.tanh(np.dot(w, phi[i - 1]) + b)
#    print(phi)
    return phi

def BACKWORD_NN(net, phi, dy):
    J = len(net)
    sigma = [0 for i in range(J)]
    sigma.append(np.array([dy - phi[J][0]]))
    dsigma = [0 for i in range(J + 1)]
    for i in reversed(range(J)):
        dsigma[i + 1] = sigma[i + 1] * (1 - phi[i + 1] ** 2)
        w, b = net[i]
        print(dsigma[i + 1], w)
        sigma[i] = np.dot(dsigma[i + 1], w)
    return dsigma

def UPDATE_WEIGHTS(net, phi, dsigma, l):
    for i in range(len(net) - 1):
        net[i][0] += l * np.outer(dsigma[i + 1], phi[i])
        net[i][1] += l * dsigma[i + 1]


if __name__ == '__main__':
    epoch = 1
    l = 0.1
    layer = 1
    node = 2
    ids = defaultdict(lambda: len(ids))
    feat_lab = list()
    net = list()
    tmp = list()
    with open('../../data/titles-en-train.labeled', 'r') as i_f:
        for line in i_f:
            y, x = line.strip().split('\t')
            words = x.lower().split()
            for word in words:
                 ids[word]

    with open('../../data/titles-en-train.labeled', 'r') as i_f:
        for line in i_f:
            line = line.split('\t')
            feat_lab.append([CREATE_FEATURES(line[1]), line[0]])

    net = CREATE_NETWORK(ids, node)
#    for i in range(layer):
#        tmp.append(np.random.uniform(-0.1, 0.1, node))
#    net.append(tmp)
    weight_0 = (np.random.rand(2,len(ids)) - .5) * .2 #(2*27190)
    b_0 = np.zeros(2)
    weight_1 = (np.random.rand(1,2) - .5)*.2
    b_1 = np.zeros(1)
    net = [[weight_0, b_0], [weight_1, b_1]]

    for i in range(epoch):
        for phi_0, y in feat_lab:
            y = int(y)
            phi = FORWORD_NN(net, phi_0)
            sigma = BACKWORD_NN(net, phi, y)
            UPDATE_WEIGHTS(net, phi, sigma, l)

    with open('network.dump', 'wb') as net_f:
        pickle.dump(net, net_f)

    with open('ids', 'wb') as ids_f:
        pickle.dump(dict(ids), ids_f)
