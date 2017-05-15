from collections import defaultdict
import math

l_1 = .85
l_2 = .15
V = 1000000
W = 0
H = 0

t_f = open('wiki-en-test.word','r')
m_f = open('model_file.txt','r')
props = defaultdict(lambda: 0.0)

for line in m_f:
    words = line.split('\t')
    props[words[0]] = float(words[1])

for line in t_f:
    line = line.lower()
    words = line.split()
    words.insert(0,'<s>')
    words.append('</s>')
    for i in range(1,len(words)):
        P1 = l_1 * props[words[i]] + (1 - l_1) / V
        P2 = l_2 * props[words[i-1] + ' ' + words[i]] + (1 - l_2) * P1
        H += -1 * math.log(P2,2)
        W += 1

print('entropy:{}'.format(H/W))
