import numpy as np
import sys
import pickle
from collections import defaultdict

ids = defaultdict(lambda:len(ids))
def create_features(x):
    _phi = [0] * len(ids)
    for word in x:
        _phi[ids[word]] += 1
    return _phi


def forward_nn(nw, phi0):
    _phi_d = dict()
    _phi_d[0] = phi0
    for i in range(len(nw)):
        w, b = nw[i]
        _phi_d[i+1] = np.tanh(np.dot(w, _phi_d[i]) + b)
    return _phi_d


def backward_nn(nw, _phid, _label):
    J = len(nw)
    delta = [0, 0, np.array([_label - _phid[J][0]])]
    delta_p = [0, 0, 0]
    for i in range(J-1, -1, -1):
        delta_p[i+1] = delta[i+1] * (1 - _phid[i+1]**2)
        w, b = nw[i]
        delta[i] = np.dot(delta_p[i+1], w)
    return delta_p


def update_weights(nw, _phi, _delta, eta):
    for i in range(len(nw)):
        w, b = nw[i]
        w += eta * np.outer(_delta[i+1], _phi[i])
        b += eta * _delta[i+1]

def init_nn_Normal():
    _nn = list()
    #1層目
    w0 = np.array([(np.random.randn(len(ids)))/10, (np.random.randn(len(ids)))/10])
    b0 = (np.random.randn(2))/10
    #2層目
    w1 = np.array([(np.random.randn(2))/10])
    b1 = (np.random.randn(1))/10
    _nn = [[w0,b0],[w1,b1]]
    return _nn


if __name__ == '__main__':

    train_data = "../../data/titles-en-train.labeled"
    # ids = defaultdict(lambda: len(ids))

    max_iter = int(sys.argv[1])
    try:
        eta = float(sys.argv[2])
    except:
        eta = 0.1
    _row = list()

    train_line_list = []
    with open(train_data) as f:
        for line in f:
            l = line.split()
            train_line_list.append(line.strip().split('\t'))
            _row.append([int(l[0]), l[1:]])

    for aline in train_line_list:
        label, sentence = int(aline[0]), aline[1]
        for w in sentence.split(' '):
            ids[w]
    
    feat_lab = np.zeros((len(ids) + 1), dtype=int)
    feat_lab_list = []
    for _, (label, sent_list) in enumerate(_row):
        # print("x:{}\ty:{}".format(x, y))
        phi = create_features(sent_list)
        feat_lab = [np.array(phi), label]
        feat_lab_list.append(feat_lab)
    
    # init random value [-0.1, 0.1] for each units
    # 1st layer
    w0 = np.array([(np.random.rand(len(ids)) - 0.5)/5, (np.random.rand(len(ids)) - 0.5)/5])
    b0 = (np.random.rand(2) - 0.5)/5
    # 2nd layer
    w1 = np.array([(np.random.rand(2) - 0.5)/5])
    b1 = (np.random.rand(1) - 0.5)/5
    # Now initialize network
    nn = [[w0,b0],[w1,b1]]

    # nn = init_nn_Normal()

    for epoch in range(max_iter):
        print("epoch {}/{}".format(epoch+1, max_iter))
        for phi_ndarray, y_label in feat_lab_list:
            phi_n = forward_nn(nn, phi_ndarray)
            delta_prime = backward_nn(nn, phi_n, y_label)
            update_weights(nn,phi_n, delta_prime, eta)
    
    from datetime import datetime
    d = datetime.now()
    model_file = 'nn_{0:%Y%m%d_%H%M%S}.pkl'.format(d)
    with open(model_file,'wb') as model_f, open('ids.dat','w') as ids_f:
        pickle.dump(nn, model_f)
        for word, count in ids.items():
            print("{}\t{}".format(word, count), file=ids_f)
    
