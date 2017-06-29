#python test-chainer-lstm.py ../../data/wiki-en-test.norm lstm_model xid.txt yid.txt my_answer_lstm.pos 50
#../../script/gradepos.pl ../../data/wiki-en-test.pos my_answer_lstm.pos

from collections import defaultdict
from chainer import *
import sys
import re
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
        return y_list

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

def test_rnn(fin_path, model_path, xid_path, yid_path, fout_path, hidden_size):
    x_ids = dict()
    with open(xid_path, "r") as fxid:
        for line in fxid:
            name, value = line.split("\t")
            x_ids[name] = int(value)
    y_ids = dict()
    with open(yid_path, "r") as fyid:
        for line in fyid:
            name, value = line.split("\t")
            y_ids[name] = int(value)
    ids_y = {i:tag for tag, i in y_ids.items()}

    rnn = RecurrentNeuralNetwork(len(x_ids), hidden_size, len(y_ids))
    serializers.load_npz(model_path, rnn)

    with open(fin_path, "r") as fin, open(fout_path, "w") as fout:
        for line in fin:
            x_list = list()
            for x in line.strip("\n").split(" "):
                x = x.lower()
                x_list.append(create_one_hot(create_ids(x, x_ids, "test"), len(x_ids)))
            y_list = rnn.predict_ylist(x_list)
            fout.write("{}\n".format(" ".join([ids_y[y] for y in y_list])))

if __name__ == "__main__":
    test_rnn(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]))
