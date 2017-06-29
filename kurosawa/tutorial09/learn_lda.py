import sys
import math
import random
from collections import defaultdict

def initialize(file_name,num_topics):
    xcorpus = []
    ycorpus = []
    xcounts = defaultdict(int)
    ycounts = defaultdict(int)
    with open(file_name) as f:
        for line in f:
            docid = len(xcorpus)
            words = line.split()
            topics = []
            for word in words:
                topic = random.randint(0,num_topics)
                topics.append(topic)
                Addcounts(word, topic, docid, 1, xcounts, ycounts)
            xcorpus.append(words)
            ycorpus.append(topics)
    return xcorpus,ycorpus,xcounts,ycounts

def Addcounts(word, topic, docid, amount, xcounts, ycounts):
    xcounts[str(topic)] += amount
    xcounts[word + '|' + str(topic)] += amount
    ycounts[str(docid)] += amount
    ycounts[str(topic) + '|' + str(docid)] += amount

def Sampleone(probs):
    z = sum(probs)
    remaining = random.uniform(0,z)
    for i in range(len(probs)):
        remaining -= probs[i]
        if  remaining <= 0:
            return i

if __name__ == '__main__':
    test_file = '../../data/wiki-en-documents.word'
    epoch = int(sys.argv[1])
    num_topics = 7
    alpha = 0.05
    beta = 0.05
    Nx = 1000000
    Ny = 7
    ll = 0
    xcorpus,ycorpus,xcounts,ycounts = initialize(test_file, num_topics)
    for e in range(epoch):
        for i in range(len(xcorpus)):
            for j in range(len(xcorpus[i])):
                x = xcorpus[i][j]
                y = ycorpus[i][j]
                Addcounts(x,y,i,-1,xcounts,ycounts)
                probs = []
                for k in range(num_topics):
                    pw = (xcounts[str(x) + '|' + str(k)]  + alpha) / (xcounts[str(k)] + alpha * Nx)
                    pt = (ycounts[str(k) + '|' + str(y)] + beta) / (ycounts[str(y)] + beta * Ny)
                    probs.append(pw * pt)
                new_y = Sampleone(probs)
                ll += math.log(probs[new_y])
                Addcounts(x,new_y, i, 1, xcounts,ycounts)
                ycorpus[i][j] = new_y
        print(ll)
    for i,j in sorted(xcounts.items()):
        if j != 0:
            print(i,j)
    for i,j in sorted(ycounts.items()):
        if j != 0:
            print(i,j)

