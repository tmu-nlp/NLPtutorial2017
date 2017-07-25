from collections import defaultdict
from pprint import pprint
import random

# unknown smoothing
lambda_coef = 0.95
lambda_unk = 1. - lambda_coef
V = 10**6

emission = defaultdict(lambda : lambda_unk/V) # initialize with lambda param.
transition = defaultdict(float)
possible_tags = dict()

model_file = 'wiki.model'
train_input = '../../data/wiki-en-train.norm_pos'


def create_trans(a, b):
    _ret = defaultdict(float)
    _ret['T,{},{}'.format(a, b)] = 1.
    return _ret


def create_emit(a, b):
    _ret = defaultdict(float)
    _ret['E,{},{}'.format(a, b)] = 1.
    if b[0].isupper():
        _ret['CAPS,{}'.format(a)] = 1.
    return _ret


def create_features(a, b):
    phi = defaultdict(float)
    for i in range(len(b) + 1): # 0..|b|
        if i == 0:
            first_tag = '<s>'
        else:
            first_tag = b[i-1]

        if i == len(b):
            next_tag = '</s>'
        else:
            next_tag = b[i]
        # phi.update(create_trans(first_tag, next_tag))
        phi['T,{},{}'.format(first_tag, next_tag)] += 1.

    for i in range(len(b)):
        # phi.update(create_emit(a[i],b[i]))
        phi['E,{},{}'.format(b[i], a[i])] += 1
        if a[i][0].isupper():
            phi['CAPS,{}'.format(b[i])] += 1
    return phi


def hmm_viterbi(w, _words):

    best_score = defaultdict(float)
    best_edge = defaultdict(str)
    best_score['0 <s>'] = 0.
    best_edge['0 <s>'] = None

    num_of_words = len(_words)

    for i in range(num_of_words):  # 0 <= i < len(word)
        phi = defaultdict(float)
        for prev_tag in possible_tags.keys():
            for next_tag in possible_tags.keys():
                if (str(i) + ' ' + prev_tag) in best_score.keys() and \
                                (prev_tag + ' ' + next_tag) in transition.keys():

                    phi.update(create_trans(prev_tag, next_tag))
                    phi.update(create_emit(next_tag, _words[i]))

                    # pprint(phi)
                    t_key = 'T,' + prev_tag+',' + next_tag
                    e_key = 'E,' + next_tag+',' + _words[i]
                    add_score = w[t_key] * (phi[t_key] + phi[e_key])

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
            t_key = 'T,' + prev_tag + ',' + next_tag
            score = best_score[str(num_of_words) + ' ' + prev_tag] + phi[t_key] * w[t_key]

            if str(num_of_words + 1) + ' ' + next_tag not in best_score.keys() \
                    or best_score[str(num_of_words + 1) + ' ' + next_tag] < score:
                best_score[str(num_of_words + 1) + ' ' + next_tag] = score
                best_edge[str(num_of_words + 1) + ' ' + next_tag] = str(num_of_words) + ' ' + prev_tag

    # Backword
    tags = []
    next_edge = best_edge[str(num_of_words + 1) + ' </s>']
    while next_edge is not None:
        pos, tag = next_edge.split(' ')
        if tag == "<s>":
            break
        tags.append(tag)
        next_edge = best_edge[next_edge]
    tags.reverse()
    return tags


# load model
with open(model_file, "r") as f:
    for line in f:
        _type, _ctx, _word, _prob = line.strip().split(' ')
        # print("type:{} ctx:{} word:{} prob:{}".format(_type, _ctx, _word, _prob))
        possible_tags[_ctx] = 1
        if _type == 'T':
            transition[_ctx + ' ' + _word] = float(_prob)
            # pprint("T: {}, p={}".format(_ctx + ' -> ' + _word, _prob))
        else:
            emission[_ctx + ' ' + _word] = float(_prob)
            # pprint("E: {}, p={}".format(_ctx + ' -> ' + _word, _prob))


# pprint(emission)
# pprint(transition)
# pprint(possible_tags)


w = defaultdict(float)

max_iter = 2
with open(train_input) as train_f:
    for _iter in range(max_iter):
        print('epoch:{}/{}'.format(_iter+1, max_iter))
        for line in train_f:
            wordtags_list = line.strip().split(' ')  # 改行コードを落としてスペースで分割
            #TODO wordtags_listはシャッフル
            random.shuffle(wordtags_list)
            X = list()
            y_prime = list()
            for wordtag in wordtags_list:
                word, tag = wordtag.split('_')
                X.append(word)
                y_prime.append(tag)
            # print(y_prime)

            y_hat = hmm_viterbi(w, X)
            phi_prime = create_features(X, y_prime)
            phi_hat = create_features(X, y_hat)

            # y_primeにのみ存在するkeyは+yprime
            # y_hatに飲み存在するものは-y_hat
            # 両方に存在するものはy_prime - y_hat

            delta_w = defaultdict(float)
            both_key = set(phi_prime.keys()) & set(phi_hat.keys())
            only_prime = set(phi_prime.keys()) - set(phi_hat.keys())
            only_hat = set(phi_hat.keys()) - set(phi_prime.keys())

            for k in both_key:
                delta_w[k] = phi_prime[k] - phi_hat[k]

            for k in only_prime:
                delta_w[k] = phi_prime[k]

            for k in only_hat:
                delta_w[k] = -phi_hat[k]

            for k, v in delta_w.items():
                w[k] += v

    import dill
    with open('hmm_percep.dill','wb') as f:
        dill.dump((w, possible_tags, transition), f)
