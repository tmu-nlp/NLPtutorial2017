#python train-perceptron.py ../../data/titles-en-train.labeled model.txt 15

from collections import defaultdict
import sys
import re
import random

def PREDICT_ONE(w, phi):
    score = 0
    for name, value in phi.items():
        if name in w:
            score += value * w[name]
    if score >= 0:
        return 1
    else:
        return -1

def CREATE_FEATURES(x):
    phi = defaultdict(lambda: 0)
    x = x.lower()
    x = re.sub(r'''["'-\/–#・(),.!?;:{}^~－（）ー〜\[\]<>＜＞［］]''', "", x)
    words = x.split()
    for word1, word2, word3 in zip(words, words[1:], words[2:]):
        phi["UNI:{}".format(word1)] += 1
        phi["BI:{} {}".format(word1, word2)] += 1
        phi["TRI:{} {} {}".format(word1, word2, word3)] += 1
    if len(words) > 0:
        phi["UNI:{}".format(words[-1])] += 1
    if len(words) > 1:
        phi["UNI:{}".format(words[-2])] += 1
        phi["BI:{} {}".format(words[-2], words[-1])] += 1
    return phi

def UPDATE_WEIGHTS(w, phi, y):
    for name, value in phi.items():
        w[name] += value * y

def train_perceptron(fin_path, model_path, epoch):
    w = defaultdict(lambda: 0)

    for i in range(epoch):
        with open(fin_path, "r") as fin:
            data = [line for line in fin]
            random.seed(i)
            random.shuffle(data)
            for line in data:
                y, x = line.strip("\n").split("\t")
                y = int(y)
                phi = CREATE_FEATURES(x)
                y_p = PREDICT_ONE(w, phi)
                if y_p != y:
                    UPDATE_WEIGHTS(w, phi, y)
    with open(model_path, "w") as fout:
        for key, value in sorted(w.items()):
            fout.write("{}\t{}\n".format(key, value))


if __name__ == "__main__":
    train_perceptron(sys.argv[1], sys.argv[2], int(sys.argv[3]))
