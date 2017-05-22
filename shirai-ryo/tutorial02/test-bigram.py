from collections import defaultdict
import math

a = 0.95
#ラムダ1のこと
b = 0.95
#ラムダ2のこと
V = 1000000
W = 0
H = 0

f1 = open("model_file.word", "r")
f2 = open("../../data/wiki-en-test.word", "r")

probs = defaultdict(float)

for line in f1:
    words = line.split('\t')
    probs[words[0]] = float(words[1])

f1.close()

for line in f2:
    words = line.split()
    words.append("<\s>")
    words.insert(0, "<s>")
    for i in range(1, int(len(words)-1)):
        P1 = a * probs[i] + (1 - a) / V
        P2 = b * probs[words[i-1] + " " + words[1]] + (1 - b) * P1
        H += math.log(P2, 2) * -1
        W += 1

f2.close()

print("entropy = " + str(H/W))
