#python train-rnn.py ../../data/wiki-en-train.norm_pos weight.txt xid.txt yid.txt 50 50 0.01
#85.84%

from collections import defaultdict
import sys
import random
import numpy
import pickle

def create_ids(feature, ids, mode):
    if mode != "test" or feature in ids:
        return ids[feature]
    else:
        return len(ids)

def create_one_hot(i, size):
    vec = numpy.zeros(size)
    if i < size:
        vec[i] = 1
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

def gradient_rnn(network, x, h, p_predict, y_correct, y_ids):
    delta = dict()
    loss = 0.0
    for name in network.keys():
        delta[name] = numpy.zeros_like(network[name])
    d_r_p = numpy.zeros_like(network["b_r"])
    for t in range(len(x) - 1, -1, -1):
        p_correct = create_one_hot(y_correct[t], len(y_ids))
        d_o_p = p_correct - p_predict[t]
        loss -= numpy.dot(p_correct, numpy.log(p_predict[t]))
        delta["w_oh"] += numpy.outer(d_o_p, h[t])
        delta["b_o"] += d_o_p
        d_r = numpy.dot(d_r_p, network["w_rh"]) + numpy.dot(d_o_p, network["w_oh"])
        d_r_p = d_r * (1 - pow(h[t], 2))
        delta["w_rx"] += numpy.outer(d_r_p, x[t])
        delta["b_r"] += d_r_p
        if t != 0:
            delta["w_rh"] += numpy.outer(d_r_p, h[t - 1])
    return delta, loss

def update_weights(network, delta, lam):
    for name in network.keys():
        network[name] += lam * delta[name]

def train_rnn(fin_path, weight_path, xid_path, yid_path, hidden_size, epoch, lam):
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

    numpy.random.seed(1)
    network = dict()
    network["w_rx"] = 0.00002 * (numpy.random.rand(hidden_size, len(x_ids)) - 0.5)
    network["w_rh"] = 0.00002 * (numpy.random.rand(hidden_size, hidden_size) - 0.5)
    network["b_r"] = numpy.zeros(hidden_size)
    network["w_oh"] = 0.00002 * (numpy.random.rand(len(y_ids), hidden_size) - 0.5)
    network["b_o"] = numpy.zeros(len(y_ids))

    for i in range(epoch):
        random.seed(i)
        random.shuffle(data)
        accum_loss = 0.0
        for x, y_correct in data:
            h, p, y_predict = forward_rnn(network, x)
            delta, loss = gradient_rnn(network, x, h, p, y_correct, y_ids)
            update_weights(network, delta, lam)
            accum_loss += loss
        print("epoch:{}/{}\tloss:{}\tlambda:{}".format(i + 1, epoch, accum_loss, lam))
        if i != 0:
            if accum_loss > prev_loss:
                lam *= 0.5
        prev_loss = accum_loss

    with open(weight_path, "wb") as fw:
        pickle.dump(network, fw)
    with open(xid_path, "w") as fxid:
        for key, value in sorted(x_ids.items()):
            fxid.write("{}\t{}\n".format(key, value))
    with open(yid_path, "w") as fyid:
        for key, value in sorted(y_ids.items()):
            fyid.write("{}\t{}\n".format(key, value))

if __name__ == "__main__":
    train_rnn(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), float(sys.argv[7]))
