import random
from collections import defaultdict
import math

def initiallize_courpus(num_topics):
    xcourpus = list()
    ycourpus = list()
    xcounts = defaultdict(int)
    ycounts = defaultdict(int)
    vocab = defaultdict(int)

    with open('../../data/wiki-en-documents.word') as f:
        for i, line in enumerate(list(f)):
            docid = i
            topics = list()
            words = line.strip().split()
            for word in words:
                topic = random.randint(0, num_topics)
                topics.append(topic)
                xcounts, ycounts = add_counts(word, topic, docid, 1, xcounts, ycounts)
                vocab[word] += 1
            xcourpus.append(words)
            ycourpus.append(topics)

    return xcourpus, ycourpus, xcounts, ycounts, vocab

def add_counts(word, topic, docid, amount, xcounts, ycounts):
    xcounts[topic] += amount
    xcounts[word+'|'+str(topic)] += amount
    if xcounts[topic] < 0 or xcounts[word+'|'+str(topic)] < 0:
        print('xcounts < 0 error')
        exit()

    ycounts[docid] += amount
    ycounts[str(topic)+'|'+str(docid)] += amount
    if ycounts[docid] < 0 or ycounts[str(topic)+'|'+str(docid)] < 0:
        print('ycounts < 0 error')
        exit()

    return xcounts, ycounts

def sample_one(probs):
    z = sum(probs)
    remaining = random.uniform(0, z)
    for i, prob in enumerate(probs):
        remaining -= prob
        if remaining <= 0:
            return i
        elif remaining > 0 and i == len(probs)-1:
            print('sample error')
            exit()

if __name__ == "__main__":
    num_topics = 20
    alpha = 0.01
    beta = 0.01
    xcourpus, ycourpus, xcounts, ycounts, vocab = initiallize_courpus(num_topics)

    for epoch in range(5):
        log_prob = 0
        for i in range(0, len(xcourpus)):
            for j in range(0, len(xcourpus[i])):
                x = xcourpus[i][j]
                y = ycourpus[i][j]
                xcounts, ycounts = add_counts(x, y, i, -1, xcounts, ycounts)
                probs = list()
                for k in range(0, num_topics):
                    p_xk = (xcounts[x+'|'+str(k)]+alpha) / (xcounts[k]+alpha*len(vocab))
                    p_kY = (ycounts[str(k)+'|'+str(y)]+beta) / (ycounts[i]+beta*num_topics)
                    probs.append(p_xk * p_kY)
                new_y = sample_one(probs)
                log_prob += math.log(probs[new_y])
                add_counts(x, new_y, i, 1, xcounts, ycounts)
                ycourpus[i][j] = new_y
        print('{}\t{}'.format(epoch+1, log_prob))
        
    # test
    for words, topics in zip(xcourpus, ycourpus):
        for word, topic in zip(words, topics):
            print('{} {}'.format(word, topic))
        print('\n')


