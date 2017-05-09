import math

from collections import defaultdict
probabilities = defaultdict(int)

with open("model_file.word") as text:
    for line in text:
        line = line.split()
        probabilities[line[0]] = float(line[1])

W = 0
unk = 0
H = 0
with open("../../data/wiki-en-test.word") as test_file:
    for line in test_file:
        words = line.split()
        words.append("</s>")
        for w in words:
            W += 1
            P = 0.05 / 1000000
            if w in probabilities:
                P += 0.95 * probabilities[w]
            else:
                unk += 1
            H += math.log(P) * -1

print("entropy = " + str(H/W))
print("coverage = " + str((W-unk)/W))
