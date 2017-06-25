#python train-chainer-nn.py ../../data/titles-en-train.labeled nn_model id.txt 2 15 0.1

from collections import defaultdict
from chainer import *
import sys
import re
import random
import numpy
import pickle

class NeuralNetwork(Chain):
    def __init__(self, input_size, hidden_size):
        super(NeuralNetwork, self).__init__(
            link1 = links.Linear(input_size, hidden_size),
            link2 = links.Linear(hidden_size, 1),
        )
    def __call__(self, x, y):
        y_predict = self.forward(x)
        return functions.mean_squared_error(y_predict, y) / 2

    def forward(self, x):
        hidden = functions.tanh(self.link1(x))
        y_predict = functions.tanh(self.link2(hidden))
        return y_predict

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

def train_nn(fin_path, model_path, id_path, hidden_size, epoch, lam):
    ids = defaultdict(lambda: len(ids))
    with open(fin_path, "r") as fin:
        for line in fin:
            y, x = line.strip("\n").split("\t")
            create_features(x, ids, "init")

    feat_lab = list()
    with open(fin_path, "r") as fin:
        for line in fin:
            y, x = line.strip("\n").split("\t")
            feat_lab.append((create_features(x, ids, "train").reshape(1, len(ids)), int(y)))

    nn = NeuralNetwork(len(ids), hidden_size)
    optimizer = optimizers.SGD(lr=lam)
    optimizer.setup(nn)

    for i in range(epoch):
        random.seed(i)
        random.shuffle(feat_lab)
        accum_loss = 0
        for phi_0, y in feat_lab:
            nn.zerograds()
            loss = nn(Variable(phi_0.astype(numpy.float32)), Variable(numpy.array([y], dtype=numpy.float32).reshape(1, 1)))
            accum_loss += loss.data
            loss.backward()
            optimizer.update()
        print("epoch:{}/{}\tloss:{}".format(i + 1, epoch, accum_loss))

    serializers.save_npz(model_path, nn)
    with open(id_path, "w") as fid:
        for key, value in sorted(ids.items()):
            fid.write("{}\t{}\n".format(key, value))


if __name__ == "__main__":
    train_nn(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]))
