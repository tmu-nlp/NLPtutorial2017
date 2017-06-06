from collections import defaultdict
import sys
from pprint import pprint

# phi = defaultdict(lambda : 0)
def create_features(x):
    '''
    params: x文
    '''
    phi = defaultdict(lambda : 0)
    words = x.strip().split(' ')
    # print(words)
    for word in words:
        phi[word.lower()] += 1
        # phi['UNI:'+word] += 1
    return phi

def predict_one(_w, _phi): # wは重みベクトルで1次元配列
    score = 0
    for name, value in _phi.items():
        if name in _w.keys():
            score += int(value) * _w[name]
    if score >= 0:
        return 1
    else:
        return -1

# def update_weights(_w, phi, y):
#     for name, value in phi.items():
#         _w[name] += value * y

# w = defaultdict(lambda :0)

import random
if __name__ == '__main__':

    # train_data_test = '../../test/03-train-input.txt'
    train_data = '../../data/titles-en-train.labeled'

    n_iter = 5
    # n_iter = int(sys.argv[1])

    w = defaultdict(lambda: 0)
    for _ in range(n_iter):
        with open(train_data) as f:
            _templine = []
            for l in f:
                _templine.append(l)

            random.shuffle(_templine)
            for ll in _templine:
                y_label, sentence = ll.strip().split('\t')
                phi = create_features(sentence)
                y_pred = predict_one(w, phi)
                # print('{}, {}'.format(type(y_pred),type(y_label)))
                if y_pred != int(y_label):
                    for name, value in phi.items():
                        w[name] += int(value) * int(y_label)
                    # update_weights(w, phi, int(y_label))
    
    for k,v in w.items():
        # if k == 'aso' or k == 'kani':
        print('{}\t{}'.format(k,v))
