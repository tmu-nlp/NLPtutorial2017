import dill
import numpy as np
from collections import defaultdict

test_input = '../../data/mstparser-en-test.dep'
weights_model = 'ans.dill'


def make_features(stack, queue):

    features = defaultdict(int)

    if len(stack) > 0 and len(queue) > 0:
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


def shift_reduce(_queue, _weights):

    stack = [[0, 'ROOT', 'ROOT']]
    heads = [-1] * (len(_queue) + 1)

    while len(_queue) > 0 or len(stack) > 1:
        features = make_features(stack, _queue)

        score_shift = predict_score(_weights['S'], features)
        score_left = predict_score(_weights['L'], features)
        score_right = predict_score(_weights['R'], features)

        if (score_shift > score_right and score_shift > score_left
                and len(_queue) > 0) or len(stack) < 2:
            stack.append(_queue.pop(0))
        elif score_left > score_right:
            heads[stack[-2][0]] = stack[-1][0]
            stack.pop(-2)
        else:
            heads[stack[-1][0]] = stack[-2][0]
            stack.pop(-1)

    return heads



data = list()
queue = list()
true_heads = list()
true_heads_list = list()

with open(test_input) as f:
    for line in f:
        if len(line) == 1:
            data.append(queue)
            queue = []
            true_heads_list.append(true_heads)
            true_heads = []
        else:
            conll = line.strip().split()
            _id = int(conll[0])
            _word = conll[1]
            _pos = conll[3]
            true_heads.append(int(conll[6]))
            # cf.append([conll[2], conll[4], conll[5], conll[7]])
            # print('{} {} {}'.format(_id, _word, _pos))
            queue.append([_id, _word, _pos])


with open(weights_model, 'rb') as f:
    weights = dill.load(f)

heads_list = []
for q in data:
    heads = shift_reduce(q, weights)
    heads_list.append(heads[1:])

import re
ptrn = re.compile(r'(?P<hoge>^(?:.+?\t){6})(\d+?)(?P<piyo>\t)')
with open(test_input) as f, open('ans.conll', 'w') as conll_out:
    sents_block = 0
    for l in f:
        if len(l) == 1:
            print('', file=conll_out)
            sents_block +=1
        else:
            conll = l.strip().split()
            _id = int(conll[0]) - 1
            pred_head = str(heads_list[sents_block][_id])
            rpl = ptrn.sub(r'\g<hoge>' + pred_head + r'\g<piyo>', l)
            print(rpl, end='', file=conll_out)


# 自分で正解率計算
sents_num = 0
whole = 0
unmatch_counter = 0
match_counter = 0

for pred, corr in zip(heads_list, true_heads_list):
    if len(pred) == len(corr):
        whole += len(pred)
        unmatch_counter += np.count_nonzero(np.array(pred) - np.array(corr))
    else:
        print('ng')

acc = 1. - unmatch_counter/whole
print('Acc: {} = {}/{}'.format(acc, whole - unmatch_counter, whole))
