from collections import defaultdict
import sys
from pprint import pprint

def create_features(sent):
    phi = defaultdict(lambda : 0)
    words = sent.strip().split(' ')
    for word in words:
        phi[word.lower()] += 1
    return phi

def predict_one(_w, _phi):
    score = 0
    for name, value in _phi.items():
        if name in _w.keys():
            score += int(value) * _w[name]
    if score >= 0:
        return 1
    else:
        return -1

import random
if __name__ == '__main__':

    train_data = '../../data/titles-en-train.labeled'

    n_iter = int(sys.argv[1])
    feat = sys.argv[2]

    out_modelfile = 'iter_{}_{}.model'.format(n_iter, feat)

    weights_history = []
    w = defaultdict(lambda: 0)

    # load sentence
    train_line_list = []
    with open(train_data) as f:
        for l in f:
            # _, sent = l.strip().split('\t')
            train_line_list.append(l.strip().split('\t'))

    # pprint(train_line_list)
    for _ in range(n_iter):
        if feat == 'shff':
            random.shuffle(train_line_list)
        for ll in train_line_list:
            y_label, sentence = ll[0], ll[1]
            phi = create_features(sentence)
            y_pred = predict_one(w, phi)
            if not y_pred == int(y_label):
                for name, value in phi.items():
                    w[name] += int(value) * int(y_label)

    with open(out_modelfile, 'w') as f:
        for k,v in w.items():
            print('{}\t{}'.format(k,v), file=f)

    print('Finish {}'.format(out_modelfile))
