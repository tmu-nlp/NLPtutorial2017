#python train-chainer-lstm.py ../../data/wiki-en-train.norm_pos lstm_model xid.txt yid.txt 50 20 0.01

from collections import defaultdict
from chainer import *
import sys
import random
import numpy
import pickle

class RecurrentNeuralNetwork(Chain):
    def __init__(self, input_size, hidden_size, output_size):
        super(RecurrentNeuralNetwork, self).__init__(
            link_xh = links.LSTM(input_size, hidden_size),
            link_hy = links.Linear(hidden_size, output_size),
        )
    def __call__(self, x_list, y_list):
        loss = Variable(numpy.zeros((), dtype=numpy.float32))
        self.link_xh.reset_state()
        for x, y in zip(x_list, y_list):
            y_predict = self.forward(Variable(numpy.array([x], dtype=numpy.float32)))
            loss += functions.softmax_cross_entropy(y_predict, Variable(numpy.array([y], dtype=numpy.int32)))
        return loss

    def forward(self, x):
        hidden = functions.tanh(self.link_xh(x))
        return self.link_hy(hidden)

    def predict_ylist(self, x_list):
        self.link_xh.reset_state()
        y_list = list()
        for x in x_list:
            y_predict = functions.softmax(self.forward(Variable(numpy.array([x], dtype=numpy.float32))))
            y_list.append(numpy.argmax(list(y_predict.data[0])))
        return y_predict

def create_ids(feature, ids, mode):
    if mode != "test" or feature in ids:
        return ids[feature]
    else:
        return len(ids)

def create_one_hot(id, size):
    vec = [0] * size
    if id < size:
        vec[id] = 1
    return vec

def softmax(x):
    return numpy.exp(x - max(x)) / sum(numpy.exp(x - max(x)))

def train_rnn(fin_path, model_path, xid_path, yid_path, hidden_size, epoch, lam):
    x_ids = defaultdict(lambda: len(x_ids))
    y_ids = defaultdict(lambda: len(y_ids))
    with open(fin_path, "r") as fin:
        for line in fin:
            wordtags = line.strip("\n").split(" ")
            for wordtag in wordtags:
                x, y = wordtag.split("_")
                x = x.lower()
                create_ids(x, x_ids, "init")
                create_ids(y, y_ids, "init")

    data = list()
    with open(fin_path, "r") as fin:
        for line in fin:
            x_list = list()
            y_list = list()
            wordtags = line.strip("\n").split(" ")
            for wordtag in wordtags:
                x, y = wordtag.split("_")
                x = x.lower()
                x_list.append(create_one_hot(create_ids(x, x_ids, "train"), len(x_ids)))
                y_list.append(create_ids(y, y_ids, "train"))
            data.append((x_list, y_list))

    rnn = RecurrentNeuralNetwork(len(x_ids), hidden_size, len(y_ids))
    optimizer = optimizers.SGD(lr=lam)
    optimizer.setup(rnn)

    for i in range(epoch):
        random.seed(i)
        random.shuffle(data)
        accum_loss = 0.0
        for x_list, y_list in data:
            rnn.zerograds()
            loss = rnn(x_list, y_list)
            accum_loss += loss.data
            loss.backward()
            optimizer.update()
        print("epoch:{}/{}\tloss:{}".format(i + 1, epoch, accum_loss))

    serializers.save_npz(model_path, rnn)
    with open(xid_path, "w") as fxid:
        for key, value in sorted(x_ids.items()):
            fxid.write("{}\t{}\n".format(key, value))
    with open(yid_path, "w") as fyid:
        for key, value in sorted(y_ids.items()):
            fyid.write("{}\t{}\n".format(key, value))

if __name__ == "__main__":
    train_rnn(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), float(sys.argv[7]))
