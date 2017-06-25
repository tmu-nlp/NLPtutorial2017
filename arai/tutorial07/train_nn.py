import numpy as np
from collections import defaultdict
import pickle

def predict_one(w, phi):
    score = np.dot(w, phi)
    if score[0] >= 0:
        return 1
    else:
        return -1


def create_features(x):
    phi0 = [0] * len(ids)
    words = x.split()
    #print(len(ids),words)
    for word in words:
        phi0[ids[word]] += 1
       
    return phi0

def forward_nn(net, phi0):
    atai = [0,0,0]
    atai[0] = phi0
    for i in range(len(net)):
        w, b = net[i]
        atai[i+1] = np.tanh(np.dot(w, atai[i] ) + b)
    #print(atai[-1])
    return atai

def backward_nn(net, atai, ydash):
    J = len(net)
    gosa = [0,0, np.array([ydash - atai[J][0]])] 
    gosadash = [0,0,0]
    #print(gosa, gosadash)
    for i in range(J-1, -1, -1):
        #print(i)
        gosadash[i+1] = gosa[i+1] * (1 - atai[i+1] ** 2)
        w, b = net[i]
        gosa[i] = np.dot(gosadash[i + 1], w)
        #print(gosa,gosadash)
    return gosadash

def update_weights(net , atai , gosadash, lambda1):
    for i in range(0, len(net)):
        w,b = net[i]
        net[i][0] += lambda1 * np.outer(gosadash[i+1], atai[i])
        net[i][1] += lambda1 * gosadash[i +1]

if __name__ == '__main__':
    ids = defaultdict(lambda : len(ids))
    l = 1
    lambda1 = 0.1
    with open('../../data/titles-en-train.labeled') as data:
        for line in data:
            y, x = line.strip().split('\t')
            words = x.split()
            for word in words:
                ids[word]
    omosa0 = np.array((np.random.rand(2, len(ids))-0.5)/5)
    bias0 = (np.random.rand(2) - 0.5) / 5
    omosa1 = np.array([(np.random.rand(2)-0.5)/5])
    bias1 = (np.random.rand(1) - 0.5) / 5
    net = [[omosa0, bias0], [omosa1, bias1]]
    for i in range(l):
        with open('../../data/titles-en-train.labeled') as data:
            for line in data:
                y, x = line.strip().split('\t')
                y = int(y)
                phi0 = create_features(x)
                atai = forward_nn(net, phi0)           
                gosadash = backward_nn(net, atai, y)
                update_weights(net, atai, gosadash, lambda1)
   
    with open ('weight_file.txt', 'wb') as w_f:
        pickle.dump(net, w_f)
    with open('id_file.txt', 'w') as i_f:
        for key,value in sorted(ids.items()):
            print('{}\t{}'.format(key, value), file = i_f)
        

