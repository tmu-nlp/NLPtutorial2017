from collections import defaultdict
from pprint import pprint
from math import log2

# unknown smoothing
lambda_coef = 0.95
lambda_unk = 1. - lambda_coef
V = 10**6


emission = defaultdict(lambda : lambda_unk/V) # initialize with lambda param.
transition = defaultdict(float)
possible_tags = dict()


model_file = 'wiki.model'


def create_trans(a, b):
    _ret = defaultdict(int)
    _ret['T,{},{}'.format(a, b)] = 1
    return _ret


def create_emit(a, b):
    _ret = defaultdict(int)
    _ret['E,{},{}'.format(a, b)] = 1
    return _ret


def hmm_viterbi(_words):

    best_score = defaultdict(float)
    best_edge = defaultdict(str)
    best_score['0 <s>'] = 0.
    best_edge['0 <s>'] = None

    num_of_words = len(_words)

    for i in range(num_of_words):  # 0 <= i < len(word)
        phi = defaultdict(int)
        for prev_tag in possible_tags.keys():
            for next_tag in possible_tags.keys():
                if (str(i) + ' ' + prev_tag) in best_score.keys() and \
                                (prev_tag + ' ' + next_tag) in transition.keys():

                    phi.update(create_trans(prev_tag, next_tag))
                    phi.update(create_emit(next_tag, _words[i]))

                    t_key = 'T,{},{}'.format(prev_tag, next_tag)
                    e_key = 'E,{},{}'.format(next_tag, _words[i])
                    add_score = w_global[t_key] * phi[t_key] + w_global[e_key] * phi[e_key]

                    score = best_score[str(i) + ' ' + prev_tag] + add_score

                    if str(i + 1) + ' ' + next_tag not in best_score.keys() \
                            or best_score[str(i + 1) + ' ' + next_tag] < score:
                        best_score[str(i + 1) + ' ' + next_tag] = score
                        best_edge[str(i + 1) + ' ' + next_tag] = str(i) + ' ' + prev_tag

    # 最後に終端記号</s>への遷移を計算する
    # i -> num_of_words
    next_tag = '</s>'
    for prev_tag in possible_tags.keys():
        if (str(num_of_words) + ' ' + prev_tag) in best_score.keys() and \
                        (prev_tag + ' ' + next_tag) in transition.keys():

            phi.update(create_trans(prev_tag, next_tag))
            t_key = 'T,{},{}'.format(prev_tag, next_tag)
            add_score = w_global[t_key] * phi[t_key]
            score = best_score[str(num_of_words) + ' ' + prev_tag] + add_score

            if str(num_of_words + 1) + ' ' + next_tag not in best_score.keys() \
                    or best_score[str(num_of_words + 1) + ' ' + next_tag] < score:
                best_score[str(num_of_words + 1) + ' ' + next_tag] = score
                best_edge[str(num_of_words + 1) + ' ' + next_tag] = str(num_of_words) + ' ' + prev_tag

    # Backword
    tags = []
    next_edge = best_edge[str(num_of_words + 1) + ' </s>']
    while next_edge is not None:
        # print(next_edge)
        pos, tag = next_edge.split(' ')
        if tag == "<s>":
            break
        tags.append(tag)
        next_edge = best_edge[next_edge]
    tags.reverse()
    return tags


import dill
test_input = '../../data/wiki-en-test.norm'
if __name__ == "__main__":

    with open('hmm_percep.dill', 'rb') as f:
        w_global, possible_tags, transition = dill.load(f)

    with open(test_input) as f, open('hmm-percep.ans', 'w') as ans_f:
        for l in f:
            words = l.strip().split()
            print(' '.join(hmm_viterbi(words)), file=ans_f)
