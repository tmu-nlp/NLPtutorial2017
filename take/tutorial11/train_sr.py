from collections import defaultdict
from pprint import pprint
import numpy as np
import dill

train_data = '../../data/mstparser-en-train.dep'
# train_data = '../../data/mstparser-en-test.dep'


def argmax_score(s: list):
    m = np.argmax(s)
    print(m)
    return ['S', 'L', 'R'][m]


def make_features(stack, queue):

    features = defaultdict(int)

    if len(stack) > 0 and len(queue) > 0:
        # print('stck-> ',stack[-1][1])
        # print('queue-> ', queue[0][1])
        features['W-1' + stack[-1][1] + ',W0' + queue[0][1]] += 1
        features['W-1' + stack[-1][1] + ',P0' + queue[0][2]] += 1
        features['P-1' + stack[-1][2] + ',W0' + queue[0][1]] += 1
        features['P-1' + stack[-1][2] + ',P0' + queue[0][2]] += 1

    if len(stack) > 1:
        features['W-2' + stack[-2][1] + ',W-1' + stack[-1][1]] += 1
        features['W-2' + stack[-2][1] + ',P-1' + stack[-1][2]] += 1
        features['P-2' + stack[-2][2] + ',W-1' + stack[-1][1]] += 1
        features['P-2' + stack[-2][2] + ',P-1' + stack[-1][2]] += 1

    return features


def predict_score(_w, _feat):
    _score = 0
    for k in _feat.keys():
        _score += _w[k] * _feat[k]
    return _score


def shift_reduce(_queue, _heads, weights):

    stack = [[0, 'ROOT', 'ROOT']]
    unproc = list()

    for i in range(len(_heads)):
        unproc.append(_heads.count(i))


    while len(_queue) > 0 or len(stack) > 1:
        features = make_features(stack, _queue)

        score_shift = predict_score(weights['S'], features)
        score_left = predict_score(weights['L'], features)
        score_right = predict_score(weights['R'], features)


        # max_score_op = argmax_score([score_shift, score_left, score_right])
        # if (max_score_op == 'S' and len(_queue) > 0) or len(stack) < 2:
        #     predict = 'S'
        # else:
        #     predict = max_score_op

        if (score_shift > score_right and score_shift > score_left and len(_queue) > 0) or len(stack) <2:
            predict = 'S'
        elif score_left > score_right:
            predict = 'L'
        else:
            predict = 'R'

        print('predcheck',predict)

        if len(stack) < 2:
            correct = 'S'
        elif _heads[stack[-1][0]] == stack[-2][0] and unproc[stack[-1][0]] == 0:
            correct = 'R'
        elif _heads[stack[-2][0]] == stack[-1][0] and unproc[stack[-2][0]] == 0:
            correct = 'L'
        else:
            correct = 'S'

        #update weights
        if not predict == correct:
            for k, v in features.items():
                if k in weights[predict].keys():
                    weights[predict][k] -= v
                if k in weights[correct].keys():
                    weights[correct][k] += v

        if correct == 'S':
            stack.append(_queue.pop(0))
        elif correct == 'L':
            unproc[stack[-1][0]] -= 1
            stack.pop(-2)
        elif correct == 'R':
            unproc[stack[-2][0]] -= 1
            stack.pop(-1)


if __name__ == '__main__':

    max_iter = 100

    data = list()
    queue = list()
    heads = [-1]
    
    with open(train_data) as f:
        for line in f:
            if len(line) == 1:
            # if not line or line.startswith('\n'):
                data.append([queue, heads])
                queue = []
                heads = [-1]
            else:
                # CoNLL形式
                # ID 単語 原型 品詞品詞2 拡張親 ラベル
                conll = line.strip().split()
                # pprint('{} conll:{} line:{}'.format(conll, len(conll), line))
                _id = int(conll[0])
                _word = conll[1]
                _pos = conll[3]
                _head = int(conll[6])
                print('{} {} {} {}'.format(_id, _word, _pos, _head))
                queue.append([_id, _word, _pos])
                heads.append(_head)

    weights = dict()
    weights['S'] = defaultdict(int)
    weights['R'] = defaultdict(int)
    weights['L'] = defaultdict(int)

    import random
    for iter in range(max_iter):
        random.shuffle(data)
        for _, (q, h) in enumerate(data):
            shift_reduce(q, h, weights)

    with open('ans.dill', 'wb') as f:
        dill.dump(weights, f)
        # print(weights, file=f)
