import math
from collections import defaultdict

probs1 = defaultdict(lambda: 0)
probs2 = defaultdict(lambda: 0)

with open('modelfile.txt', 'r') as f1:
    for line in f1:
        w = line.split()
        probs2[w[0], w[1]] = w[2]
with open('../tutorial01/modelfile.txt', 'r') as f2:
    for line in f2:
        w = line.split()
        probs1[w[0]] = w[1]
V = 1000000
H = 0
W = 0
r1 = 0
r2 = 0
for j in range(19):
    r2 += 0.05
    for i in range(19):
        r1 += 0.05
        H = 0
        W = 0
        with open('../../data/wiki-en-test.word', 'r') as f3:
            for line in f3:
                word = line.split()
                word.append('</s>')
                word.insert(0,'<s>')
                for k in range(1, len(word) - 1):
                    P1 = float(r1) * float(probs1[word[k-1]]) + float((1 - r1) / V)
                    P2 = float(r2) * float(probs2[word[k-1], word[k]]) + float((1 - r2) * P1)
                    H += -math.log2(P2)
                    W += 1

            print('r1={} r2={} entrophy={}' .format(r1, r2, H / W))
