from train-nn import forward_rnn
import numpy as np
import pickle

with open ('weight_file.dump', 'rb') as w_file:
    net = pickle.load(w_file)
with open ('x_id_file.dump', 'rb') as id_file:
    ids = pickle.load(id_file)

def Create_one_hot(word, size)
    vec = np.zeros(sizse)
    if word in ids:
        vec[ids[0][word]] = 1
    return vec

with open ('my_answer.txt', 'w') as text:
    for line in open('../../data/wiki-en-test.norm'):
        x_list = list()
        words = line.split()
        for x in words:
            x_list.append(Create_one_hot(x, len(ids[0])))
        h, p, y = forword_rnn(net[0], net[1], net[2], net[3], net[4], x_list)
        new_dict = {}
        for key, value in ids[1].items():
            new_dict[value] = key
        ans = []
        for i in y:
            ans.append(new_dict[i])
        text.write(" ".join(ans) + "\n")

