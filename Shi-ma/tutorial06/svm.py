import collections
import math
import random

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

def CREATE_FEATURES(txt):
    phi = collections.defaultdict(lambda: 0)
    words = txt.split()
    for word in words:
        phi['UNI:' + word] += 1
    return phi

def PREDICT_ONE(w, phi, calc=False):
    score = 0
    for name, value in phi.items():
        if name in w.keys():
            score += value*w[name]
    if calc == 1:
        return score
    else:
        if score >= 0:
            return 1
        else:
            return -1

def UPDATE_WEIGHTS(w, phi, y, i):
    last = collections.defaultdict(lambda: 0)
    c = 10**(-4)
    for name, value in phi.items():
        GET_W(w, name, c, i, last)
        w[name] += value*y

def GET_W(w, name, c, i, last):
    if i != last[name]:
        c_size = c * (i - last[name])
        if abs(w[name]) <= c_size:
            w[name] = 0
        else:
            w[name] -= sign(w[name]) * c_size
        last[name] = i
    return w[name]

def train_svm(data_train, w, i):
    margen = 45
    for line in data_train:
        y, txt = line.split('\t')
        y = int(y)
        txt = txt.lower()
        phi = CREATE_FEATURES(txt)
        val = PREDICT_ONE(w, phi, calc=True)*y
        if val <= margen:
            UPDATE_WEIGHTS(w, phi, y, i)

def train_svm_epoch(epoch, path_data_train):
    w = collections.defaultdict(lambda: 0)
    for i in range(epoch):
        with open(path_data_train, 'r') as data_train:
            data_train_list = list(data_train)
            random.shuffle(data_train_list)
            train_svm(data_train_list, w, i)
    return w

if __name__ == '__main__':
    epoch = 10
    path_data_train = '../../data/titles-en-train.labeled'
    w = train_svm_epoch(epoch, path_data_train)

    with open('../../data/titles-en-test.word', 'r') as data_test:
        with open('my_answer.txt', 'w') as data_out:
            for txt in data_test:
                txt = txt.lower()
                phi = CREATE_FEATURES(txt)
                y_predict = PREDICT_ONE(w, phi)
                print(y_predict, file=data_out)
