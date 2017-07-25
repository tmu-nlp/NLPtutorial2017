import random
import numpy as np
from collections import defaultdict


def ADDCOUNTS(w, t, d_id, amount):
    t = str(t)

    xcounts[t] += amount
    xcounts[w + '|' + t] += amount

    ycounts[d_id] += amount
    ycounts[t + '|' + str(d_id)] += amount

def PROB_x(x, k):
    return (xcounts[x + '|' + str(k)] + alpha) / (xcounts[str(k)] + alpha * N_x)

def PROB_y(k, y):
    return (ycounts[str(k) + '|' + str(y)] + beta) / (ycounts[str(y)] + beta * N_y)

def SAMPLEONE(prob):
    z = sum(prob)
    remaining = random.random() * z
    for i in range(len(prob)):
        remaining -= prob[i]
        if remaining <= 0:
            return i
    print('error',prob)
    print(z)


if __name__ == '__main__':
    train = '../../data/wiki-en-documents.word'
    test = '../../test/07-train.txt'
    NUM_TOPICS = 2
    epoch = 10
    with open(test, 'r') as train_f:
        xcorpus = list()
        ycorpus = list()
        xcounts = defaultdict(lambda: 0)
        ycounts = defaultdict(lambda: 0)
        for line in train_f:
            docid = len(xcorpus)
            words = line.split()
            topics = list()
            for word in words:
                topic = int(random.uniform(0, NUM_TOPICS))
                topics.append(topic)
                ADDCOUNTS(word, topic, docid, 1)
            xcorpus.append(words)
            ycorpus.append(topics)
#    print(xcorpus)
#    print(ycorpus)
    alpha = .01
    beta = .01
    N_x = 1000 #N_x数えるの手間だったので見逃していただきたく
    N_y = NUM_TOPICS
    pp = 0
    for l in range(epoch):
        print('epoch', l)
        for i in range(len(xcorpus)):
            for j in range(len(xcorpus[i])):
                x = xcorpus[i][j]
                y = ycorpus[i][j]
                ADDCOUNTS(x, y, i, -1)
                probs = list()
                for k in range(NUM_TOPICS):
                    probs.append(PROB_x(x, k) * PROB_y(k, i))
                new_y = SAMPLEONE(probs)
                pp += np.log(probs[new_y])
                ADDCOUNTS(x, new_y, i, 1)
                ycorpus[i][j] = new_y
                print(xcorpus[i][j], ycorpus[i][j])
'''
    with open('../../test/07-train.txt', 'r') as f:
        result = defaultdict(list)
        for id_x, line in enumerate(f):
            words = line.split()
            for word in words:
                probs = list()
                for k in range(NUM_TOPICS):
                    probs.append(PROB_x(word, k) * PROB_y(k, id_x))
                predict = SAMPLEONE(probs)
                result[predict].append(word)
        print(result)
'''
