import sys
import math
from collections import defaultdict

r1 = 0.95
r2 = 0.95
V = 1000000
W = 0
H = 0


w_prob = defaultdict(lambda: 0)
t_cnt = 0

with open('bigram_model.txt','r') as f:
#with open(sys.argv[1], 'r') as f:
    for line in f:
#        line = line.lower()

        w_list = line.split('\t')
        w_prob[w_list[0]] = float(w_list[1])

#print(w_prob.items())

with open(sys.argv[1], 'r') as f2:
    for line in f2:
        tw_list = line.split()
        tw_list.append('</s>')
        tw_list.insert(0,"<s>")
        for i in range(1,len(tw_list)):
            p1 = r1 * w_prob[tw_list[i]] + ( 1 - r1 )/V
            P2 = r2 * w_prob[tw_list[i-1] + " " + tw_list[i]] + (1 - r2) * p1

            H += -math.log(P2,2)
            W += 1
print()
print("r1 = {}, r2 = {}" .format(r1, r2) )
print("entropy = {}" .format( H/W ))
print()
