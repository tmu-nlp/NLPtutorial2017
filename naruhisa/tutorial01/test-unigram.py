from collections import defaultdict
import numpy as np

probability = defaultdict()

with open('modelfile.txt', 'r') as f1:
    for line in f1:
        line1 = line.split()
        probability[line1[0]] = line1[1]

r = 0.95
r_unk = 1 - r
V = 1000000
W = 0
H = 0
unk = 0

with open('wiki-en-test.word', 'r') as f2:
    for line in f2:
        line = line.lower()
        words = line.split()
        words.append('</s>')
        for w in words:
            W += 1
            P = r_unk / float(V)
            if w in probability:
                P += float(r) * float(probability[w])
            else:
                unk += 1
            H += -np.log2(P)
print('entrophy = ', H / W)
print('coverage = ', (float(W) - float(unk)) / float(W))
