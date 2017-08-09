#python test-hmm-perceptron.py ../../data/wiki-en-test.norm weights.txt ids.txt my_answer.pos
#../../script/gradepos.pl ../../data/wiki-en-test.pos my_answer.pos

from collections import defaultdict
import sys
import random
import numpy
import pickle

def init_ids(X, Y, ids, possible_tags):
    for i in range(len(Y) + 1):
        if i == 0:
            first_tag = "<s>"
        else:
            first_tag = Y[i - 1]
        if i == len(Y):
            next_tag = "</s>"
        else:
            next_tag = Y[i]
        ids["T,{},{}".format(first_tag, next_tag)]
        if i != len(Y):
            ids["E,{},{}".format(Y[i], X[i])]
            ids["CAPS,{}".format(Y[i])]
            possible_tags[Y[i]]

def create_features(X, Y, ids):
    phi = numpy.zeros(len(ids))
    for i in range(len(Y) + 1):
        if i == 0:
            first_tag = "<s>"
        else:
            first_tag = Y[i - 1]
        if i == len(Y):
            next_tag = "</s>"
        else:
            next_tag = Y[i]
        phi += create_transition(first_tag, next_tag, ids)
        if i != len(Y):
            phi += create_emission(Y[i], X[i], ids)
    return phi

def create_transition(first_tag, next_tag, ids):
    phi = numpy.zeros(len(ids))
    if "T,{},{}".format(first_tag, next_tag) in ids:
        phi[ids["T,{},{}".format(first_tag, next_tag)]] = 1
    return phi

def create_emission(y, x, ids):
    phi = numpy.zeros(len(ids))
    if "E,{},{}".format(y, x) in ids:
        phi[ids["E,{},{}".format(y, x)]] = 1
    if "CAPS,{}".format(y) in ids:
        phi[ids["CAPS,{}".format(y)]] = 1
    return phi


def hmm_viterbi(weights, X, ids, possible_tags):
    best_score = dict()
    best_edge = dict()
    best_score["0 <s>"] = 0
    best_edge["0 <s>"] = "NULL"
    
    for i in range(0, len(X)):
        for prev in possible_tags.keys():
            if "{} {}".format(i, prev) in best_score:
                for nex in possible_tags.keys():
                    score = best_score["{} {}".format(i, prev)] + numpy.dot(weights, create_transition(prev, nex, ids) + create_emission(nex, X[i], ids))
                    if "{} {}".format(i + 1, nex) not in best_score or best_score["{} {}".format(i + 1, nex)] < score:
                        best_score["{} {}".format(i + 1, nex)] = score
                        best_edge["{} {}".format(i + 1, nex)] = "{} {}".format(i, prev)
    for prev in possible_tags.keys():
        if "{} {}".format(len(X), prev) in best_score:
            score = best_score["{} {}".format(len(X), prev)] + numpy.dot(weights, create_transition(prev, "</s>", ids))
            if "{} </s>".format(len(X) + 1) not in best_score or best_score["{} </s>".format(len(X) + 1)] < score:
                best_score["{} </s>".format(len(X) + 1)] = score
                best_edge["{} </s>".format(len(X) + 1)] = "{} {}".format(len(X), prev)

    tags = list()
    next_edge = best_edge["{} </s>".format(len(X) + 1)]
    while next_edge != "0 <s>":
        position, tag = next_edge.split()
        tags.append(tag)
        next_edge = best_edge[next_edge]
    tags.reverse()

    return tags

def test_hmm_perceptron(fin_path, weight_path, id_path, fout_path):
    possible_tags = defaultdict(lambda: 0)
    possible_tags["<s>"]
    
    with open(id_path, "rb") as fid:
        ids = pickle.load(fid)

    for key in ids.keys():
        if key[:5] == "CAPS,":
            possible_tags[key[5:]]

    with open(weight_path, "rb") as fw:
        weights = pickle.load(fw)

    with open(fin_path, "r") as fin, open(fout_path, "w") as fout:
        for line in fin:
            X = line.strip("\n").split(" ")
            Y_hat = hmm_viterbi(weights, X, ids, possible_tags)
            fout.write("{}\n".format(" ".join(Y_hat)))

if __name__ == "__main__":
    test_hmm_perceptron(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
