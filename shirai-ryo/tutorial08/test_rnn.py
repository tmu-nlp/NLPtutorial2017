from train_rnn import forward_rnn
import pickle
import numpy as np

with open('weight_file.dump', 'rb') as w_file:
    net = pickle.load(w_file)
with open('id_file.dump', 'rb') as id_file:
    ids = pickle.load(id_file)


def CREATE_ONE_HOT(word, size):
    vec = np.zeros(size)
    if word in ids:
        vec[ids[0][word]] = 1
    return vec


with open('my_answer.txt', 'w') as text:
    for line in open('../../data/wiki-en-test.norm', 'r'):
        x_list = list()
        words = line.split()
        for x in words:
            x_list.append(CREATE_ONE_HOT(x, len(ids[0])))
        h, p, y = forward_rnn(net[0], net[1], net[2], net[3], net[4], x_list)
        new_dict = {}
        for key, value in ids[1].items():
            new_dict[value] = key
        ans = []
        for i in y:
            ans.append(new_dict[i])
        text.write(" ".join(ans) + "\n")
