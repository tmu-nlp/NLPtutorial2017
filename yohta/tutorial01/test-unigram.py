from collections import defaultdict
import math

lambda_1 = 0.95
lambda_unk = 1 - lambda_1
V = 1000000
W = 0
H = 0
unk = 0
probabilities = defaultdict(int)
#probabilities = dict()

m_f = open('train-model.txt','r')
t_f = open('wiki-en-test.word','r')

for line in m_f:
    words = line.split()
    probabilities[words[0]] = float(words[1])
#    print(words)

for line in t_f:
    line = line.lower()
    words = line.split()
    words.append('<\s>')
    for word in words:
        W += 1
        P = lambda_unk / V
        if word in probabilities:
            P += lambda_1 * probabilities[word]
        else:
            unk += 1
        H += (-1 * math.log(P,2))

print('entropy:{}'.format(H/W))
print('coverage:{}'.format((W - unk)/ W))
