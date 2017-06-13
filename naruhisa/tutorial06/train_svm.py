from collections import defaultdict
import numpy as np

train_f = '../../data/titles-en-train.labeled'
model_f = 'model_file.txt'

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
    words = x.strip().split()
    for word in words:
        phi[word] += 1
    return phi

def UPDATE_WEIGHTS(w, k, y, ite, v):
    w[k] += v * float(y)
    ite[k] += 1


def sign(x):
    if x > 0:
        return 1
    else:
        return -1

def getw(w, name, c, ite, last):
    if ite[name] > last[name]:
        c_size = c * (ite[name] - last[name])
        if abs(w[name]) <= c_size:
            w[name] = 0
        else:
            w[name] -= sign(w[name]) * c_size
        last[name] = ite[name]
    return w[name]

if __name__ == '__main__':
    w = defaultdict(lambda: 0)
    ite = defaultdict(lambda: 0)
    last = defaultdict(lambda: 0)
    epoch = 25
    margin = 10
    c = 0.001
    for i in range(epoch):
        with open('../../data/titles-en-train.labeled', 'r') as f:
            for line in f:
                y, x = line.strip().split('\t')
                phi = defaultdict(lambda: 0)
                for k, v in CREATE_FEATURES(x).items():
                    phi[k] = int(v)
                    val = getw(w, k, c, ite, last) * phi[k] * int(y)

                    if val <= margin:
                        UPDATE_WEIGHTS(w, k, y, ite, v)
    with open(model_f, 'w') as f_m:
        for k, v in sorted(w.items(), key = lambda x:x[1]):
            f_m.write('{} {}\n'.format(k, getw(w, k, c, ite, last)))
