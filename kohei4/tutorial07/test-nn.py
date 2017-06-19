from collections import defaultdict
import numpy as np
import math
import sys
import pickle

begin = 0
end = 0

data = []



def create_features(x):
    phi = [0 for i in range(len(ids))]
    for word in x:
        if 'UNI:'+word in ids:
            phi[ids["UNI:"+word]] += 1
    return phi

def forward_nn(net,phi0):
    phi_d = dict()
    phi_d[0] = phi0
    for ii in range(len(net)):
        w, b = net[ii]
        phi_d[ii+1] = np.tanh(np.dot(w,phi_d[ii]) + b)

    return phi_d

with open('net','rb') as ff:
    nn = pickle.load(ff)
#print(nn)

with open('ids','r') as gg:
    ids = dict()
    for line in gg:
        line = line.strip().split('\t')
        ids[line[0]] = int(line[1])
#print(ids)
#print(len(ids))

test_input = "../../data/titles-en-test.word"

with open(test_input,'r') as f:
    for ii, line in enumerate(f) :
        #if ii >= begin and ii <= end:
            words = line.split()
            phi0 = create_features(words)
            #print(phi0)
            phi = forward_nn(nn,phi0)
            y_dush = (1 if phi[len(nn) - 1][0] >= 0 else -1)
            print(y_dush)
