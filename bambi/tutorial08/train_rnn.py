import numpy as np
import pickle
import math
from collections import defaultdict

def softmax(x):
    # https://stackoverflow.com/questions/34968722/softmax-function-python
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def create_one_hot(id_target, size):
    vec = np.zeros(size)
    vec[id_target] = 1
    return vec

def create_ids(filename):
    x_ids = defaultdict(lambda: len(x_ids))
    y_ids = defaultdict(lambda: len(y_ids))
    for line in open(filename):
        wordtags = line.strip("\n").split(" ")
        for wordtag in wordtags:
            word, tag = wordtag.split("_")
            x_ids[word]
            y_ids[tag]
    return x_ids, y_ids

def create_feature_labels(filename, x_ids, y_ids):
    feature_labels = []
    for line in open(filename):
        wordtags = line.strip("\n").split(" ")
        x_list = []
        y_list = []
        for wordtag in wordtags:
            word, tag = wordtag.split("_")
            X = create_one_hot(x_ids[word],len(x_ids))
            x_list.append(X)
            Y = create_one_hot(y_ids[tag],len(y_ids))
            y_list.append(Y)
        feature_labels.append((x_list,y_list))
    return feature_labels

def find_max(p):
    y = 0
    for i in range(len(p)):
        if p[i] > p[y]:
            y = i
    return y

def forward_rnn(wRx,wRh,br,wOh,bo,x):
    h = [0] * len(x) # hidden layers at time
    p = [0] * len(x) # output prob at time t
    y = [0] * len(x) # output value at time t
    for t in range(len(x)):
        if t > 0:
            h[t] = np.tanh(wRx.dot(x[t]) + wRh.dot(h[t-1]) + br)
        else:
            h[t] = np.tanh(wRx.dot(x[t]) + br)
        p[t] = softmax(wOh.dot(h[t]) + bo)
        #y[t] = np.argmax(p[t])
        y[t] = find_max(p[t])
    return h, p, y

def gradient_rnn(wRx,wRh,br,wOh,bo,x,h,p,y_):
    del_wRx = [0]
    del_wRh = [0]
    del_br = [0]
    del_wOh = [0]
    del_bo = [0] # initialize

    phiR_ = np.zeros(len(br)) #  Error from the following time step
    for t in range(len(x)):
        phi0_ = p[t] - y_[t]# Output error
        del_wOh += np.outer(phi0_,h[t])
        phiR = np.dot(phiR_,wRh) + np.dot(phi0_,wOh) # Backprop
        phiR_ = phiR * (1- pow(h[t],2)) # tanh gradient
        del_wRx += np.outer(phiR_,x[t])
        del_bo += phi0_ # Δb0 += δ0' Output gradient
        del_br += phiR_ # Δbr += δr' Hidden gradient
        if t != 0:
            del_wRh += np.outer(phiR_,h[t-1])
    return del_wRx, del_wRh, del_br, del_wOh, del_bo

def update_weights(wRx,wRh,br,wOh,bo,del_wRx,del_wRh,del_br,del_wOh,del_bo,lamb):
    wRx += lamb * del_wRx
    wRh += lamb * del_wRh
    br += lamb * del_br
    wOh += lamb * del_wOh
    bo += lamb * del_bo
    return wRx,wRh,br,wOh,bo

if __name__ == "__main__":
    #file = "../test/05-train-input.txt"
    file = "../data/wiki-en-train.norm_pos"

    x_ids, y_ids = create_ids(file)
    feature_labels = create_feature_labels(file,x_ids,y_ids)
    wRx = (np.random.rand(2, len(x_ids)) - 0.5)/5 # (2,vocab)
    wOh =(np.random.rand(len(y_ids), 2) - 0.5) /5 # (pos_type,2)
    wRh = (np.random.rand(2,2) - 0.5) /5 #(2,2)
    bo = np.zeros(len(y_ids))
    br = np.zeros(2)
    epoch = 3
    lamb = 0.015
    for _ in range(epoch):
        for x, y_correct in feature_labels:
            h,p,y_predict = forward_rnn(wRx,wRh,br,wOh,bo,x)
            #print(y_predict)
            del_wRx, del_wRh, del_br, del_wOh, del_bo = gradient_rnn(wRx,wRh,br,wOh,bo,x,h,p,y_correct)
            wRx,wRh,br,wOh,bo, = update_weights(wRx,wRh,br,wOh,bo,del_wRx,del_wRh,del_br,del_wOh,del_bo,lamb)

    net = (wRx,wRh,br,wOh,bo)

    pickle.dump(net, open("net.pickle", "wb" ) )
    with open("x_ids.txt","w") as output:
        for k,v in x_ids.items():
            print("{}\t{}".format(k,v), file=output)
    with open("y_ids.txt","w") as output:
        for k,v in y_ids.items():
            print("{}\t{}".format(k,v), file=output)
