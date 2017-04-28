from collections import defaultdict

lambda_1 = 0.95
lambda_unk = 1 - lambda_1
V = 1000000
W = H = 0
unk = 0

# GEN_MODEL_FILE = "sample.model"
GEN_MODEL_FILE = "wiki.model"

# TEST_FILE = "01-test-input.txt"
TEST_FILE = "../data/wiki-en-test.word"

""" load model """
probability = defaultdict(lambda: 0)
with open(GEN_MODEL_FILE) as f:
    for line in f:
        l = line.strip('\n').split()
        probability[l[0]] = float(l[1])

import math
with open(TEST_FILE) as f:
    for line in f:
        l = line.split()
        l.append('</s>')

        for w in l:
            W += 1
            p = lambda_unk/V
            if w in probability:
                p += lambda_1 * probability[w]
            else:
                unk += 1
            H += - math.log2(p)

print('entropy:{}'.format(H/W))
print('coverage{}:'.format((W-unk)/W))