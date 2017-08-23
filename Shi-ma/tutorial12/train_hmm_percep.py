from collections import defaultdict, Counter
import random
import dill



def CREATE_TRANS(first_tag, next_tag):
    phi_trans = defaultdict(lambda: 0)
    phi_trans[(first_tag, next_tag)] += 1

    return phi_trans



def CREATE_EMIT(y, x):
    phi_emit = defaultdict(lambda: 0)
    phi_emit[(y, x)] += 1
    phi_emit[y] += 1

    return phi_emit



def CREATE_FEATURES(X, Y):
    phi = Counter()
    for i in range(len(Y) + 1):
        if i == 0:
            first_tag = '<s>'
        else:
            first_tag = Y[i-1]
        if i == len(Y):
            next_tag = '</s>'
        else:
            next_tag = Y[i]
        phi.update(CREATE_TRANS(first_tag, next_tag))

    for i in range(len(Y)):
        phi.update(CREATE_EMIT(Y[i], X[i]))

    return phi



def PREDICT(w, phi):
    score = 0
    for key, value in phi.items():
        score += w[key] * value

    return score



def HMM_VITERBI(w, X, possible_tags, transition):
    best_score = dict()
    best_score[(0, '<s>')] = 0
    best_edge = dict()
    best_edge[(0, '<s>')] = None

    for i in range(len(X)):
        for prev in possible_tags:
            for next_ in possible_tags:
                if (i, prev) in best_score and (prev, next_) in transition:
                    score = best_score[(i, prev)] + PREDICT(w, CREATE_TRANS(prev, next_)) + PREDICT(w, CREATE_EMIT(next_, X[i]))
                    if (i+1, next_) not in best_score or best_score[(i+1, next_)] < score:
                        best_score[(i+1, next_)] = score
                        best_edge[(i+1, next_)] = (i, prev)
    i = len(X)
    for prev in possible_tags:
        next_ = '</s>'
        if (i, prev) in best_score:
            score = best_score[(i, prev)] + PREDICT(w, CREATE_TRANS(prev, next_)) + PREDICT(w, CREATE_EMIT(next_, '</s>'))
            if (i+1, next_) not in best_score or best_score[(i+1, next_)] < score:
                best_score[(i+1, next_)] = score
                best_edge[(i+1, next_)] = (i, prev)

    Y_hat = []
    next_edge = best_edge[(len(X)+1, '</s>')]
    while next_edge != (0, '<s>'):
        Y_hat.append(next_edge[1])
        next_edge = best_edge[next_edge]
    Y_hat.reverse()

    return Y_hat



def train(data_train, w, possible_tags, transition):
    for line in data_train:
        word_pos_list = line.strip().split()
        X, Y_prime = list(map(lambda x: x.split('_')[0], word_pos_list)), list(map(lambda x: x.split('_')[1], word_pos_list))
        Y_hat = HMM_VITERBI(w, X, possible_tags, transition)

        phi_prime = CREATE_FEATURES(X, Y_prime)
        phi_hat = CREATE_FEATURES(X, Y_hat)

        for key in set(phi_prime) | set(phi_hat):
            w[key] += phi_prime[key] - phi_hat[key]



def train_epoch(epoch, path_data_train, path_w_p_t_out):
    w = defaultdict(lambda: 0)
    # w = defaultdict(lambda: 0)
    with open(path_data_train) as data_train:
        data_train_list = list(data_train)

    possible_tags = set()
    transition = set()
    for line in data_train_list:
        preb_pos = '<s>'
        possible_tags.add(preb_pos)
        for pos in map(lambda x: x.split('_')[1], line.strip().split()):
            possible_tags.add(pos)
            transition.add((preb_pos, pos))
            preb_pos = pos
        transition.add((pos, '</s>'))
    possible_tags.add('</s>')

    data_train_shuffled = data_train_list[:]
    for i in range(epoch):
        print('{} epoch'.format(i+1))
        random.seed(i)
        # random.shuffle(data_train_shuffled)
        train(data_train_shuffled, w, possible_tags, transition)
    with open(path_w_p_t_out, 'wb') as w_p_t_out:
        dill.dump([w, possible_tags, transition], w_p_t_out)



if __name__ == '__main__':
    epoch = 5
    path_data_train = '../../data/wiki-en-train.norm_pos'
    # path_data_train = '../../test/05-train-input.txt'
    path_w_p_t_out = 'result/w_p_t.dump'
    train_epoch(epoch, path_data_train, path_w_p_t_out)
