#python train-nn.py ../../data/titles-en-train.labeled weight.txt id.txt 2 25 0.1

from collections import defaultdict
import sys
import re
import random
import numpy
import pickle

def create_features(x, ids, mode):
    if mode == "init":
        phi = defaultdict(lambda: 0)
    elif mode == "train" or mode == "test":
        phi = numpy.zeros(len(ids))
    x = x.lower()
    x = re.sub(r'''["'-\/–#・(),.!?;:{}^~－（）ー〜\[\]<>＜＞［］]''', "", x)
    words = x.split()
    for word in words:
        if mode != "test" or "UNI:{}".format(word) in ids:
            phi[ids["UNI:{}".format(word)]] += 1
    return phi

def forward_nn(network, phi_0):
    phi = [phi_0]
    for w, b in network:
        phi.append(numpy.tanh(numpy.dot(w, phi[-1]) + b))
    return phi

def backward_nn(network, phi, y):
    J = len(network)
    delta = [0] * (J + 1)
    delta[-1] = numpy.array([y - phi[J][0]])
    delta_p = [0] * (J + 1)
    for i in range(J - 1, -1, -1):
        delta_p[i + 1] = delta[i + 1] * (1 - pow(phi[i + 1], 2))
        delta[i] = numpy.dot(delta_p[i + 1], network[i][0])
    return delta_p

def update_weights(network, phi, delta_d, lam):
    for i in range(len(network)):
        w, b = network[i]
        w += lam * numpy.outer(delta_d[i + 1], phi[i])
        b += lam * delta_d[i + 1]

def train_nn(fin_path, weight_path, id_path, hidden_size, epoch, lam):
    ids = defaultdict(lambda: len(ids))
    with open(fin_path, "r") as fin:
        for line in fin:
            y, x = line.strip("\n").split("\t")
            create_features(x, ids, "init")

    feat_lab = list()
    with open(fin_path, "r") as fin:
        for line in fin:
            y, x = line.strip("\n").split("\t")
            feat_lab.append((create_features(x, ids, "train"), int(y)))

    numpy.random.seed(1)
    network = [tuple()] * 2
    network[0] = (0.2 * numpy.random.rand(hidden_size, len(ids)) - 0.1, numpy.zeros(hidden_size))
    network[1] = (0.2 * numpy.random.rand(1, hidden_size) - 0.1, numpy.zeros(1))

    for i in range(epoch):
        random.seed(i)
        random.shuffle(feat_lab)
        accum_loss = 0.0
        for phi_0, y in feat_lab:
            phi = forward_nn(network, phi_0)
            delta_p = backward_nn(network, phi, y)
            update_weights(network, phi, delta_p, lam)
            accum_loss += ((y - phi[-1][0]) ** 2) / 2
        print("epoch:{}/{}\tloss:{}\tlambda:{}".format(i + 1, epoch, accum_loss, lam))
        if i != 0:
            if accum_loss > prev_loss:
                lam *= 0.5
        prev_loss = accum_loss

    with open(weight_path, "wb") as fw:
        pickle.dump(network, fw)
    with open(id_path, "w") as fid:
        for key, value in sorted(ids.items()):
            fid.write("{}\t{}\n".format(key, value))


if __name__ == "__main__":
    train_nn(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]))
