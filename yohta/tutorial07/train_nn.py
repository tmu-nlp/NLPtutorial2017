from collections import defaultdict
import numpy as np
import random
import pickle

ids = defaultdict(lambda :len(ids))
# phi = defaultdict(lambda :len(ids))

def create_features(x): # UNI:word --> word_id
    # create list phi (len = len(ids))
    phi = np.zeros(len(ids))
    for word in x:
        # Training
        phi[ids["UNI:" + word]] += 1
        # Testing
        # if "UNI:" + word in ids:
        # phi[ids["UNI:" + word]] += 1
    return phi

def create_ids(x):
    for word in x:
        ids["UNI:" + word]


#def predict_one(w,phi):
#    score = np.dot(w,phi)


def forward_nn(net,phi_0):
#    phi = [0] * len(ids)
    phi = [phi_0] # insert layer_feature
    for i in range(len(net)):
        w,b = net[i]
        phi.append(np.tanh(np.dot(w,phi[i]) + b))
#        print(phi[i]) # --> complete
    return phi

def backward_nn(net,phi,y_):
    j = len(net)
    delt = [0] * j
    delt.append(np.array([y_ - phi[j][0]]))
    delt_ = [0] * (j+1)
    for i in reversed(range(j)):
        delt_[i+1] = delt[i+1] * (1 - phi[i+1]**2)
        w,b = net[i]
        delt[i] = np.dot(delt_[i+1],w)
#        print(delt_) # --> complete(?)
    return delt_


def update_weights(net,phi,delt_,lam):
    for i in range(len(net)):
        w,b = net[i]
        net[i][0] += lam * np.outer(delt_[i+1],phi[i])
        net[i][1] += lam * delt_[i+1]
#        print('w:{}\tb:{}\n'.format(w,b)) # --> complete


if __name__ == '__main__':
    l = 5
    lam = 0.1
    #array = defaultdict(lambda :0)
    feat_lab = []
    data = []
    print('epoch:{}'.format(l))
    with open('../../data/titles-en-train.labeled') as train:
        for line in train:
            y,x = line.split('\t') # x = words,y = default_feature
            data.append([y,x])

        for y,x in data:
            x = x.split() # x:words --> x:word_list
            create_ids(x) # count demension and replace word to word_id

        for y,x in data:
            x = x.split()
            y = int(y)
            phi_0 = create_features(x) # phi_0 = list
#              print(phi_0) # --> complete (in this for)
            feat_lab.append([y,phi_0])
        random.shuffle(feat_lab)
#        print(feat_lab) # --> complete

    # initialize net randomly
        nn = []
        w_0 = (np.random.rand(2,len(ids)) - 0.5) * 2 # -1~1
        b_0 = np.zeros(2)
        w_1 = (np.random.rand(1,2) - 0.5) * 2
        b_1 = np.zeros(1)
        net = [[w_0,b_0],[w_1,b_1]]
#        print(network) # --> complete (?)

        for i in range(l): # (l = iterations)
#            err = 0
            for y,phi_0 in feat_lab:
                phi = forward_nn(net,phi_0)
                delt_ = backward_nn(net,phi,y)
                update_weights(net,phi,delt_,lam)
    with open('weight_file.txt','wb') as we_f,open('id_file.txt','wb') as id_f:
        pickle.dump(net,we_f)
        pickle.dump(dict(ids),id_f)
