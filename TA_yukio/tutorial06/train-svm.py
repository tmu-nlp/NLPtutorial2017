#python train-svm.py ../../data/titles-en-train.labeled model.txt 10 22.0 0.01

from collections import defaultdict
import sys
import re
import random

def PREDICT_SCORE(w, phi):
    score = 0.0
    for name, value in phi.items():
        if name in w:
            score += value * w[name]
    return score

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

def getw(w, phi, c, i, last):
    for name in phi.keys():
        if i != last[name]:
            c_size = c * (i - last[name])
            if abs(w[name]) <= c_size:
                w[name] = 0
            else:
                w[name] -= c_size * w[name]
                #w[name] -= c_size if w[name] > 0 else -c_size
            last[name] = i

def train_svm(fin_path, model_path, epoch, margin, c):
    w = defaultdict(lambda: 0.0)
    last = defaultdict(lambda: 0)

    for i in range(epoch):
        with open(fin_path, "r") as fin:
            data = [line for line in fin]
            random.seed(i)
            random.shuffle(data)
            for line in data:
                y, x = line.strip("\n").split("\t")
                y = int(y)
                phi = CREATE_FEATURES(x)
                val = PREDICT_SCORE(w, phi) * y
                if val <= margin:
                    getw(w, phi, c, i, last)
                    UPDATE_WEIGHTS(w, phi, y)
    with open(model_path, "w") as fout:
        for key, value in sorted(w.items()):
            fout.write("{}\t{}\n".format(key, value))


if __name__ == "__main__":
    train_svm(sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))
