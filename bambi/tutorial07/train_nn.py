
import numpy as np
import pickle
from collections import defaultdict

def create_feature(x, ids):
    phi = np.zeros(len(ids))
    for word in x.split():
        k = ids["UNI:"+word]
        phi[k] += 1
    return phi

def forward_nn(net, phi_zero):
    N = len(net) + 1
    phi = [0] * N
    phi[0] = phi_zero
    for i in range(len(net)):
        w,b = net[i]
        a = np.dot(w, phi[i])
        phi[i+1] = np.tanh(a + b)
    return phi

def update_weight(net,phi,delta_,lamb):
    for i in range(len(net)):
        w,b = net[i]
        w += lamb * np.outer(delta_[i+1], phi[i])
        b += lamb * delta_[i+1]

def backward_nn(net, phi, y_):
    J = len(net)
    y = np.array(y_, dtype=float)
    b = np.array([y - phi[J][0]])
    delta = []
    for _ in range(J):
        delta.append(0)
    delta.append(b)

    delta_ = []
    for _ in range(len(delta)):
        delta_.append(0)
    for i in range(J-1,-1,-1):
        delta_[i + 1] = delta[i + 1] * (1 - (phi[i + 1] ** 2))
        w, b = net[i]
        a = np.dot(delta_[i + 1], w)
        delta[i] = a
    return delta_

if __name__ == "__main__":
    #file = "../test/03-train-input.txt"
    file = "../../data/titles-en-train.labeled"
    feat_lab = list()
    ids = defaultdict(lambda: len(ids))
    for line in open(file):
        y,x = line.strip("\n").split("\t")
        for i in x.split():
            ids["UNI:"+i] # default dict will handle id value automatically (auto-increment)

    for line in open(file):
        y,x = line.strip("\n").split("\t")
        feat_lab.append((create_feature(x, ids),y))

    I = 10
    lamb = 0.1
    seed = np.random.rand(len(ids))-0.5
    w0 = np.array([seed,seed])# (2,vocab)
    w1 = np.array([np.random.rand(2)-0.5]) #(1,2)
    b0 = np.random.rand(2)-0.5
    b1 = np.random.rand(1)-0.5
    net = np.array([[w0,b0],[w1,b1]])
    for _ in range(I):
        for phi_zero, y in feat_lab:
            phi = forward_nn(net, phi_zero)
            delta_ = backward_nn(net, phi, y)
            update_weight(net,phi,delta_,lamb)

    pickle.dump(net, open("weight.pickle", "wb" ) )
    # pickle cannot dump dict easily (lazy to find out)
    with open("ids.txt","w") as output:
        for k,v in ids.items():
            print("{}\t{}".format(k,v), file=output)
