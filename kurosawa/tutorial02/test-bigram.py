from collections import defaultdict
import sys
import math

lambda_1 = 0.85
lambda_2 = 0.35
V = 1000000
W = 0
H = 0
dic = defaultdict(int)
with open('model.txt') as model:
    for line in model:
        line = line.split('\t')
        dic[line[0]] = line[1]

with open(sys.argv[1]) as data:
    for words in data:
        words = words.lower().split()
        words.insert(0,'<s>')
        words.append('</s>')
        for i in range(1,len(words)):
            P1 = lambda_1 * float(dic[words[i]]) + (1 - lambda_1) / V
            P2 = lambda_2 * float(dic[words[i-1]+' '+words[i]]) + (1-lambda_2) * P1
            H += -1 * math.log2(P2)
            W += 1
    print('entropy : {}'.format(H/W))
