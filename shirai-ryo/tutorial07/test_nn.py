from train_nn import forward_nn
import numpy as np
import pickle

with open('weight_file.dump', 'rb') as w_file:
    network = pickle.load(w_file)
with open('id_file.dump', 'rb') as id_file:
    ids = pickle.load(id_file)

with open('my_answer.txt', 'w') as text:
    for line in open('../../data/titles-en-test.word', 'r'):
        features = np.zeros(len(ids))
        for word in line.strip('\n').split():
            if word in ids:
                    features[ids[word]] += 1
        polarity = 1 if forward_nn(network, features)[2][0] > 0 else -1
        text.write(str(polarity) + '\t' + line)

# .strip('\n')
