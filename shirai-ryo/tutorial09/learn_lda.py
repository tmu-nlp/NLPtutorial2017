import random
from collections import defaultdict
import math

def SAMPLEONE(probs):
    Z = sum(probs)
    remaining = random.uniform(0,Z)
    # random.uniform(x,y)… x～yまでのfloat値を取得します。
    for i in range(len(probs)):
        remaining -= probs[i]
        if remaining <= 0:
            return i
    print('error')
    exit()

def ADDCOUNTS(word, topic, docid, amount):
    x_counts[topic] += amount
    x_counts[word + "|" + str(topic)] += amount
    y_counts[docid] += amount
    y_counts[str(topic) + "|" + str(docid)] += amount



# 実装：初期化
if __name__ == "__main__":
    with open("../../test/07-train.txt", "r") as train_text:
        x_corpus = []
        y_corpus = []
        x_counts = defaultdict(int)
        y_counts = defaultdict(int)
        N = 2
        NX = defaultdict(int)
        for line in train_text:
            docid = len(x_corpus)
            words = line.split()
            topics = []
            for word in words:
                topic = random.randint(0, N)
                topics.append(topic)
                ADDCOUNTS(word, topic, docid, 1)
                NX[word] += 1
            x_corpus.append(words)
            y_corpus.append(topics)

    alpha = 0.01
    beta = 0.01
    epoch = 10
    for e in range(epoch):
        ll = 0
        for i in range(len(x_corpus)):
            for j in range(len(x_corpus[i])):
                x = x_corpus[i][j]
                y = y_corpus[i][j]
                ADDCOUNTS(x, y, i, -1)
                probs = []
                for k in range(N):
                    probs.append(((x_counts[x + '|' +str(k)]+alpha) / (x_counts[k]+alpha*len(NX))) * ((y_counts[str(k) + '|' + str(i)] + beta) / (y_counts[i] + beta * N)))
                new_y = SAMPLEONE(probs)
                print(new_y)
                ll += math.log(probs[new_y])
                ADDCOUNTS(x, new_y, i, 1)
                y_corpus[i][j] = new_y
        print(ll)

        print(x_corpus)
        print(y_corpus)


"""



"""
