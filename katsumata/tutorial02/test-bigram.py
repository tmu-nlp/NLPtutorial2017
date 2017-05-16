from collections import defaultdict
import sys
import math
probs = defaultdict(lambda : .0)
"""
lambda1 = .85
lambda2 = .15
"""
V = 10 ** 6
W = 0
H = 0
entropy = 100.0 

with open('model_file.txt', 'r') as m_f:
    for row in m_f:
        words = row.strip('\n').split(':')
        probs[words[0]] = words[1]
"""
線形探索でlambda1,2を求める方
"""
with open(sys.argv[1], 'r') as t_f:
    contex = list()
    for line in t_f:
        line = line.lower()
        words = line.split()
        words.append('</s>')
        words.insert(0, '<s>')
        contex.append(words)
for lambda1 in range(5,100,5):
    for lambda2 in range(5,100,5):
        H = 0
        W = 0
        for words in contex:
            for i in range(1, len(words)-1):
                bigram = ' '.join(words[i-1:i+1])
                P1 = lambda1*.01 * float(probs[words[i]]) + (1 - lambda1*.01) / V
                P2 = lambda2*.01 * float(probs[bigram]) + (1 - lambda2*.01) * P1
                H += -1 * math.log2(P2)
                W += 1
        print('entoropy:{}'.format(H/W))
        print('lambda1 : {}\nlambda2 : {}'.format(lambda1*.01,lambda2*.01))
        if (H / W) < entropy:
            entropy = H / W
            good_lambda1 = lambda1*.01
            good_lambda2 = lambda2*.01
print ('\n After all......\n')
print ('entropy = {}'.format(entropy))
print ('lambda1 = {}, lambda2 = {}'.format(good_lambda1,good_lambda2))
"""
適当にlambda1,2決めてやるやつ
"""
"""
with open(sys.argv[1], 'r') as t_f:
    for line in t_f:
        line = line.lower()
        words = line.split()
        words.append('</s>')
        words.insert(0, '<s>')
        for i in range(1, len(words)):
            temp_str1 = ' '.join(words[i-1:i+1])
            print (temp_str1)
            P1 = lambda1 * float(probs[words[i]]) + (1 - lambda1) / V
            P2 = lambda2 * float(probs[temp_str1]) + (1 - lambda2) * P1
            H += -1 * math.log2(P2)
            W += 1
print ('entropy = {}'.format(H/W))
"""
