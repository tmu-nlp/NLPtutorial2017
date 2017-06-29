from train_rnn import train_rnn
from train_rnn import forward_rnn
import numpy as np
from collections import defaultdict
import random

def test_rnn():
    hidden_layer, output_layer, x_ids, y_ids = train_rnn()
    vocab_size = len(x_ids)
    output_size = len(y_ids)
    with open("../../data/wiki-en-test.norm") as f:
        for line in f:
            h_list = list()
            tag_list = list()
            word_list = line.strip().lower().split()
            for t, word in enumerate(word_list):
                x_vector = create_onehot_for_test(word, x_ids) 
                h_list, _, output = forward_rnn(hidden_layer, output_layer, x_vector, h_list, t)
                tag_list.append(id_to_tag(output, y_ids))
            print(' '.join(tag_list))    

def id_to_tag(tag_id, y_ids):
    for tag, value in y_ids.items():
        if tag_id == value:
            return tag

def create_onehot_for_test(word, x_ids):
    x_vector = np.zeros((1, len(x_ids)))
    if word in x_ids:
        x_vector[0][x_ids[word]] = 1
    
    return x_vector
    

if __name__ == "__main__":
   test_rnn() 

