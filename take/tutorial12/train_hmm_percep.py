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

train_input = '../../data/wiki-en-train.norm_pos'


def create_trans(a, b):
    _ret = defaultdict(int)
    _ret['T,{},{}'.format(a, b)] = 1
    return _ret


def create_emit(a, b):
    _ret = defaultdict(int)
    _ret['E,{},{}'.format(a, b)] = 1
    return _ret


def create_features(a, b):
    phi = defaultdict(int)
    for i in range(len(b) + 1):
        if i == 0:
            first_tag = '<s>'
        else:
            first_tag = b[i-1]

        if i == len(b):
            next_tag = '</s>'
        else:
            next_tag = b[i]
        # phi.update(create_trans(first_tag, next_tag))
        phi['T,{},{}'.format(first_tag, next_tag)] += 1

    for i in range(len(b)):
        # phi.update(create_emit(a[i],b[i]))
        phi['E,{},{}'.format(b[i], a[i])] += 1
    return phi


def hmm_viterbi(_words):

    best_score = defaultdict(int)
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

                    # pprint(phi)
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
        pos, tag = next_edge.split(' ')
        if tag == "<s>":
            break
        tags.append(tag)
        next_edge = best_edge[next_edge]
    tags.reverse()
    return tags


if __name__ == '__main__':

    # load model; given by tutorial04
    model_file = 'wiki.model'
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

    # global scope
    w_global = defaultdict(int)
    
    max_iter = 1
    with open(train_input) as train_f:
        for _iter in range(max_iter):
            print('epoch:{}/{}'.format(_iter+1, max_iter))
            for line in train_f:
                wordtags_list = line.strip().split(' ')  # 改行コードを落としてスペースで分割
                random.shuffle(wordtags_list)
                X = list()
                y_prime = list()
                for wordtag in wordtags_list:
                    word, tag = wordtag.split('_')
                    X.append(word)
                    y_prime.append(tag)
    
                y_hat = hmm_viterbi(X)
                phi_prime = create_features(X, y_prime)
                phi_hat = create_features(X, y_hat)
    
                delta_w = defaultdict(int)
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
                    w_global[k] += v
    
        import dill
        with open('hmm_percep.dill','wb') as f:
            dill.dump((w_global, possible_tags, transition), f)
