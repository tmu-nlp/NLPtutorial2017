from collections import defaultdict
import numpy as np
import random
import math

NUM_TOPIC = 20
xcounts = defaultdict(int)
ycounts = defaultdict(int)

def addcounts(word, topic, doc_id, amount):
    xcounts[str(topic)] += amount
    xcounts[word + " " + str(topic)] += amount
    ycounts[str(doc_id)] += amount
    ycounts[str(topic) + " " + str(doc_id)] += amount

#file = "../test/07-train.txt"
file = "../../data/wiki-en-documents.word"
xcorpus = []
ycorpus = []
for line in open(file):
    doc_id = len(xcorpus)
    words = line.split()
    topics = []
    for word in words:
        topic = str(random.randint(0, NUM_TOPIC))
        topics.append(topic)
        addcounts(word,topic,doc_id,1)
    xcorpus.append(words)
    ycorpus.append(topics)

def sampleOne(probs):
    z = sum(probs)
    remaining = random.uniform(0,z)
    for i in range(len(probs)):
        remaining -= probs[i]
        if remaining <= 0:
            return i

epoch = 2
lamb = 0.2
for _ in range(epoch):
    ll = 0
    for i in range(len(xcorpus)):
        for j in range(len(xcorpus[i])):
            x = xcorpus[i][j]
            y = ycorpus[i][j]
            addcounts(x,y,i,-1)
            probs = []
            for k in range(NUM_TOPIC):
                # to prevent division by zero problem, add lambda
                p_xlk = xcounts[x + " " + str(k)] + lamb /(xcounts[str(k)] + lamb)
                p_kly = ycounts[str(k)+ " " + str(i)] + lamb/(ycounts[i] + lamb)
                probs.append(p_xlk * p_kly)
            new_y = sampleOne(probs)
            ll += math.log(probs[new_y])
            addcounts(x,new_y,i,1)
            ycorpus[i][j] = new_y
    print(ll)

for x,y in zip(xcorpus,ycorpus):
    for w,t in zip(x,y):
        print(w,t)
