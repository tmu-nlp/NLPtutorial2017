import random
from collections import defaultdict
import sys
import math
from pprint import pprint

JOINT = '_'


def add_counts(word:str, topic:str, docid:str, amount:int):
    xcounts[topic] += amount
    xcounts[word + JOINT + topic] += amount

    ycounts[docid] += amount
    ycounts[topic + JOINT + docid] += amount

    assert xcounts[topic] >= 0
    assert xcounts[word + JOINT + topic] >= 0
    assert ycounts[docid] >= 0
    assert ycounts[topic + JOINT + docid] >= 0


def sample_one(probs:list):
    z = sum(probs)
    remaining = random.random() * z
    
    for i, p in enumerate(probs):
        remaining -= p
        if remaining <= 0:
            return i

    raise Exception('sample_one:なんかおかしい')


if __name__ == '__main__':

    NUM_TOPICS = 2
    MAX_ITER = 10000
    alpha = 2
    beta = 2

    inputfile = '../../test/07-train.txt'
    try:
        if sys.argv[1] == 'r':
            inputfile = '../../data/wiki-en-documents.word'
    except:
        pass

    xcorpus = list()
    ycorpus = list()
    xcounts = defaultdict(lambda: 0)
    ycounts = defaultdict(lambda: 0)

    _nx = set()

    with open(inputfile) as f:
        for line in f:
            docid = len(xcorpus)
            words = line.lower().strip().split()
            topics = list()
            for word in words:
                _nx.add(word)
                topic = random.randint(0, NUM_TOPICS-1)
                topics.append(topic)
                add_counts(word, str(topic), str(docid), 1)
            xcorpus.append(words)
            ycorpus.append(topics)

    Nx = len(_nx) 
    Ny = NUM_TOPICS

    for curr_iter in range(MAX_ITER):
        print('Iter {}/{}'.format(curr_iter+1, MAX_ITER))
        ll = 0
        for i in range(len(xcorpus)):
            for j in range(len(xcorpus[i])):
                x = xcorpus[i][j]
                y = ycorpus[i][j]
                add_counts(x, str(y), str(i), -1)
                probs = list()
                for k in range(NUM_TOPICS):
                    p_xk = (xcounts[x+JOINT+str(k)] + alpha)/(xcounts[str(k)] + alpha * Nx)
                    p_kY = (ycounts[str(k)+JOINT+str(i)] + beta)/(ycounts[str(i)] + beta * Ny)
                    probs.append(p_xk * p_kY)
                new_y = sample_one(probs)
                ll += math.log(probs[new_y])
                add_counts(x, str(new_y), str(i), 1)
                ycorpus[i][j] = new_y
            print(ll)

    pprint(sorted(xcounts.items(), key=lambda x: x[0]))
    pprint(sorted(ycounts.items(), key=lambda x: x[0]))

