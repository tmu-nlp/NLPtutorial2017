# coding: utf-8
from collections import defaultdict
import numpy as np
import math
import sys
import pickle

n_iter = 1
ramda = 0.1

begin = 0
end = 10

data=[]
data2=[]

ids = defaultdict(lambda:len(ids))
def create_ids(x):
    #global ids
    #ids = defaultdict(lambda:len(ids))
    for word in x:
        ids["UNI:"+word]


def create_features(x):
    phi = [0 for i in range(len(ids))]
    for word in x:
        phi[ids["UNI:"+word]] += 1
    return phi

def init_nn():
    nn = []
    w0 = np.array([(np.random.rand(len(ids))-0.5)/5, (np.random.rand(len(ids))-0.5)/5])
    b0 = (np.random.rand(2)-0.5)/5
    w1 = np.array([(np.random.rand(2)-0.5)/5])
    b1 = (np.random.rand(1)-0.5)/5
    nn = [[w0,b0],[w1,b1]]
    #print('w0shape={}'.format(w0.shape))
    #print('w1shape={}'.format(w1.shape))

    return nn

def forward_nn(net,phi0):
    phi_d = dict()
    phi_d[0] = phi0
    for ii in range(len(net)):
        w, b = net[ii]
        #w = net[ii][0]
        #b = net[ii][1]
        #print(phi_d[ii].shape)
        #print(w.shape)
        #print(b.shape)
        #print((np.dot(w,phi_d[ii])+b).shape)
        phi_d[ii+1] = np.tanh(np.dot(w,phi_d[ii]) + b)

    return phi_d

def backward_nn(net,phi_d,y_l):
    J = len(net)
    delta = [0,0,np.array([y_l - phi_d[J][0]])]
    #print(phi_d[J][0])
    #print(delta.shape,delta)
    delta_d = [0,0,0]
    #print(delta_d.shape,delta_d)
    for i in range(J-1,-1,-1):
        #print(i)
        delta_d[i+1] = delta[i+1]*(1 - pow(phi_d[i+1],2) )
        #print(delta_d[i+1].shape,delta_d[i +1 ])
        w, b = net[i]
        #print('w_shape_w={},{}'.format(w.shape,w))
        #print('delta_d.w={}'.format(np.dot(delta_d[i+1],w)))
        delta[i] = np.dot(delta_d[i+1],w)
    return delta_d

def update_weight(net,phi_d,delta_d,ramda):
    for i in range(len(net)):
        w,b = net[i]
        w += ramda*np.outer(delta_d[i+1],phi_d[i])
        b += ramda*delta_d[i+1]



#train_input = "../../data/titles-en-train.labeled"
train_input = "../../data/titles-en-train.labeled"

with open(train_input,'r') as f:
    for line in f:
        words = line.split()
        #print(words[1:])
        data.append([int(words[0]),words[1:]])

for ii, [y, x] in enumerate(data):
        create_ids(x)
#print(len(ids))
#print('n_data={}'.format(len(data)))

#make feat_lab
feat_lab = np.zeros((len(ids) +1), int)
feat_lab_list = []
for ii, [y, x] in enumerate(data):
#    if ii >= begin and ii <= end:
        phi = create_features(x)
        #feat_lab = np.array([phi,y])
        feat_lab = [np.array(phi),y]
        feat_lab_list.append(feat_lab)
#print(*feat_lab_list)

nn = init_nn()
#print('nn_init={}\n{}'.format(nn[0],nn[1]))

for ii in range(n_iter):
    for phi_array, y in feat_lab_list:
        #print(phi_array)
        #print(y)
        phi_n = forward_nn(nn,phi_array)
        #print('phi_n={}'.format(phi_n))
        delta_d = backward_nn(nn,phi_n,y)
        #rint(delta_d)
        update_weight(nn,phi_n,delta_d, ramda)

#print('nn_updated={}\n{}'.format(nn[0],nn[1]))

with open('net','wb') as ff:
    pickle.dump( nn, ff)

with open('ids','w') as gg:
    for key,value in ids.items():
        print("{}\t{}".format(key, value),file=gg)
