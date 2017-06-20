#python test-chainer-nn.py ../../data/titles-en-test.word nn_model id.txt my_answer_chainer.word
#../../script/grade-prediction.py ../../data/titles-en-test.labeled my_answer_chainer.word 2

from collections import defaultdict
from chainer import *
import sys
import re
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
        return functions.mean_squared_error(y_predict, y)

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

def test_nn(fin_path, model_path, id_path, fout_path, hidden_size):
    ids = dict()
    with open(id_path, "r") as fid:
        for line in fid:
            name, value = line.split("\t")
            ids[name] = int(value)

    nn = NeuralNetwork(len(ids), hidden_size)
    serializers.load_npz(model_path, nn)

    with open(fin_path, "r") as fin, open(fout_path, "w") as fout:
        for x in fin:
            phi_0 = create_features(x, ids, "test").reshape(1, len(ids))
            y = nn.forward(Variable(phi_0.astype(numpy.float32)))
            fout.write("{}\n".format(1 if y.data >= 0 else -1))

if __name__ == "__main__":
    test_nn(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
