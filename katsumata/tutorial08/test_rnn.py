import pickle
from train_nn import forward_rnn 
import numpy as np
"""
def create_features(x):
    phi = [0 for i in range(len(ids))]
    words = x.strip().split()
    for word in words:
        if 'UNI:'+word in ids:
            phi[ids['UNI:'+word]] += 1
    return phi        
"""
def id_to_pos(num):
    return rev_yids[num]
    
def create_onehot(arg, ids):
    vec = np.zeros(len(ids))
    if arg in ids:
        vec[ids[arg]] = 1
    #unkは[0]*len(ids)で表現している
    return vec    

if __name__ == '__main__':
    test_f = '../../data/wiki-en-test.norm'
    answer = 'my_answer.rnn' 
    with open('network.dump', 'rb') as net_f:
        net = pickle.load(net_f)
    with open('x_ids.dump', 'rb') as ids_f1:
        x_ids = pickle.load(ids_f1)
    with open('y_ids.dump', 'rb') as ids_f2:
        y_ids = pickle.load(ids_f2)
    rev_yids = dict()
    for key, value in y_ids.items():
        rev_yids[value] = key
    with open(answer, 'w') as ans_f, open(test_f, 'r') as t_f:
        for line in t_f:
            x_list = list()
            for x in line.split():
                x_list.append(create_onehot(x, x_ids))
            h, p, y_list = forward_rnn(net, x_list) #y_list:予測したposのインデックスが各単語分のリストになってる
            y_predicts = map(id_to_pos, y_list) 
            ans_f.write (' '.join(y_predicts))
            ans_f.write('\n')    
