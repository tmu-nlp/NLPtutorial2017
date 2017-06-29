#python test-rnn.py ../../data/wiki-en-test.norm weight.txt xid.txt yid.txt my_answer.pos
#../../script/gradepos.pl ../../data/wiki-en-test.pos my_answer.pos

from collections import defaultdict
import sys
import numpy
import pickle

def create_ids(feature, ids, mode):
    if mode != "test" or feature in ids:
        return ids[feature]
    else:
        return len(ids)

def create_one_hot(id, size):
    vec = numpy.zeros(size)
    if id < size:
        vec[id] = 1
    return vec

def softmax(x):
    return numpy.exp(x - max(x)) / sum(numpy.exp(x - max(x)))

def forward_rnn(network, x_list):
    h = list()
    p = list()
    y = list()
    for x in x_list:
        if len(h) > 0:
            h.append(numpy.tanh(numpy.dot(network["w_rx"], x) + numpy.dot(network["w_rh"], h[-1]) + network["b_r"]))
        else:
            h.append(numpy.tanh(numpy.dot(network["w_rx"], x) + network["b_r"]))
        p.append(softmax(numpy.dot(network["w_oh"], h[-1]) + network["b_o"]))
        y.append(numpy.argmax(p[-1]))
    return h, p, y

def test_rnn(fin_path, weight_path, xid_path, yid_path, fout_path):
    x_ids = dict()
    y_ids = dict()
    with open(xid_path, "r") as fxid:
        for line in fxid:
            name, value = line.split("\t")
            x_ids[name] = int(value)
    with open(yid_path, "r") as fyid:
        for line in fyid:
            name, value = line.split("\t")
            y_ids[name] = int(value)
    with open(weight_path, "rb") as fw:
        network = pickle.load(fw)
    ids_y = {i:tag for tag, i in y_ids.items()}

    with open(fin_path, "r") as fin, open(fout_path, "w") as fout:
        for line in fin:
            x_list = list()
            for x in line.strip("\n").split(" "):
                x = x.lower()
                x_list.append(create_one_hot(create_ids(x, x_ids, "test"), len(x_ids)))
            h, p, y_list = forward_rnn(network, x_list)
            fout.write("{}\n".format(" ".join([ids_y[y] for y in y_list])))

if __name__ == "__main__":
    test_rnn(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
