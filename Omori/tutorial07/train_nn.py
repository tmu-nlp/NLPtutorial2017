import numpy as np
from collections import defaultdict
import random
import pickle

def train_nn(train_file):
    with open(train_file) as f:
        layer_middle = 2
        f = list(f)
        network, ids = init_network(f, layer_middle)
        vocab_size = len(ids)
        for I in range(5):
            random.shuffle(f) 
            for line in f:
                label, sent = line.strip().split('\t')
                sent = sent.lower()  # w/o lower --> 88.735388% : w lower --> 93.942614%
                label = int(label)
                phi0 = create_features(sent, ids)
                phi_vector = dict_to_vector(phi0, vocab_size)
                phi = forward_nn(network, phi_vector)
                delta_ = backward_nn(network, phi, label)
                network = update_weights(network, phi, delta_, lr=0.01)
        print('... training finish ...')
    
    with open('network.pkl', mode='wb') as f:
        pickle.dump(network, f)
    with open('ids.txt', mode='w') as f:
        for word, word_id in sorted(ids.items()):
            f.write('{}\t{}\n'.format(word, word_id))

def dict_to_vector(phi, vocab_size):
    phi_vector = np.zeros((vocab_size, 1))
    for word_id, value in phi.items():
        phi_vector[word_id][0] = value    
    return phi_vector


def init_network(f, layer_middle): 
    ids = defaultdict(lambda: len(ids))
    for line in f:
        label, sent = line.strip().split('\t')
        sent = sent.lower()
        words = sent.split()
        for word in words:
            ids[word]
  
    vocab_size = len(ids)
    network = [(np.random.uniform(-0.1, 0.1, (vocab_size, layer_middle)), np.zeros((layer_middle, 1))), (np.random.uniform(-0.1, 0.1, (layer_middle, 1)), np.zeros((1,1)))]  # network=[(w_1,b_1), (w_2,b_2)]
    return network, ids

def create_features(sent, ids):
    phi = defaultdict(int)
    words = sent.split()
    for word in words:
        phi[ids[word]] += 1
    return phi

def forward_nn(network, phi0):
    phi = [[],[],[]]
    phi[0] = phi0
    for i, _ in enumerate(network):
        w, b = network[i]
        phi[i+1] = np.tanh(np.dot(w.T, phi[i])+b)
    return phi    

def backward_nn(network, phi, label):
    J = len(network)  # J is the number of layers
    delta = [[], [], []]
    delta_ = [[], [], []]  # delta_[0] is still []   
    y = phi[J]  # predict y.shape = (1,1)
    delta[J] = label - y
    for i in range(J-1, -1, -1):
        delta_[i+1] = delta[i+1] * (1 - pow(phi[i+1], 2))
        w, b = network[i]
        delta[i] = np.dot(w, delta_[i+1])
    return delta_

def update_weights(network, phi, delta_, lr):
    for i, layer in enumerate(network):
        w, b = network[i] 
        w += lr * np.outer(delta_[i+1], phi[i]).T
        b += lr * delta_[i+1]
        network[i] = (w, b)
    return network

if __name__ == "__main__":
    train_nn('../../data/titles-en-train.labeled')
    #test_nn('../../data/titles-en-test.word', network, ids)




