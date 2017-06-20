#python test-nn.py ../../data/titles-en-test.word weight.txt id.txt my_answer.word
#../../script/grade-prediction.py ../../data/titles-en-test.labeled my_answer.word

from collections import defaultdict
import sys
import re
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

def test_nn(fin_path, weight_path, id_path, fout_path):
    ids = dict()
    with open(id_path, "r") as fid:
        for line in fid:
            name, value = line.split("\t")
            ids[name] = int(value)
    with open(weight_path, "rb") as fw:
        network = pickle.load(fw)

    with open(fin_path, "r") as fin, open(fout_path, "w") as fout:
        for x in fin:
            phi_0 = create_features(x, ids, "test")
            phi = forward_nn(network, phi_0)
            fout.write("{}\n".format(1 if phi[-1][0] >= 0 else -1))

if __name__ == "__main__":
    test_nn(sys.argv[1], sys.argv[2], sys.argv[3],sys.argv[4])
