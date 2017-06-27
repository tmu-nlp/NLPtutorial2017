import numpy as np
from collections import defaultdict
import pickle

def test_nn(test_file, network, ids):
    with open(test_file) as f:
        for line in f:
            sent = line.strip()
            sent = sent.lower()
            phi0 = create_features(sent, ids)
            phi_vector = dict_to_vector(phi0, len(ids))
            phi = forward_nn(network, phi_vector)
            print (1 if phi[len(network)][0][0] >= 0 else -1)


def dict_to_vector(phi, vocab_size):
    phi_vector = np.zeros((vocab_size, 1)).astype(np.float32)
    for word_id, value in phi.items():
        phi_vector[word_id][0] = value    
    return phi_vector

def create_features(sent, ids):
    phi = defaultdict(int)
    words = sent.split()
    for word in words:
        if word in ids:  # test add
            phi[ids[word]] += 1
    return phi

def forward_nn(network, phi0):
    phi = [[],[],[]]
    phi[0] = phi0
    for i, _ in enumerate(network):
        w, b = network[i]
        phi[i+1] = np.tanh(np.dot(w.T, phi[i]))+b
    return phi    


if __name__ == "__main__":
    with open('ids.txt', mode='r') as f:
        ids = defaultdict(int)
        for line in f:
            word, word_id = line.strip().split()
            ids[word] = int(word_id)
    with open('network.pkl', mode='rb') as f:
        network = pickle.load(f)
    test_nn('../../data/titles-en-test.word', network, ids)










