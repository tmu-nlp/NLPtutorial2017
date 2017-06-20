from collections import defaultdict
from train_nn import forward_nn
import numpy as np
import pickle

ids = defaultdict(int)

def create_feature(x, ids):
    phi = np.zeros(len(ids))
    for word in x.split():
        if "UNI:" + word in ids:
            phi[ids["UNI:"+word]] += 1
    return phi

for line in open("ids.txt"):
    name, value = line.split("\t")
    ids[name] = int(value)
net = pickle.load(open( "weight.pickle", "rb" ))
'''
for each x in the data
    φ0 = create_features(x)
    φ= forward_nn(net, φ0 )
    y'= (1 if φ[len(net) - 1][0] >= 0 else -1)
'''
with open("my_answer.word", "w") as output:
    for line in open("../../data/titles-en-test.word"):
        phi0 = create_feature(line,ids)
        phi = forward_nn(net,phi0)
        y_ = (1 if phi[len(net) - 1][0] >= 0 else -1)
        print(y_,file=output)
