import sys
import math
from collections import defaultdict

r1 = 0.95
r_unk = 1-r1
V = 1000000
W = 0
H = 0
unk = 0

w_prob = defaultdict(lambda: 0)
t_cnt = 0

with open('model_file.txt','r') as f:
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
        for w in tw_list:
            W += 1
            P = r_unk / V
            if w_prob[w]:
                P += r1 * w_prob[w]
            else:
                unk += 1
            H += -math.log(P,2)

print("entropy = {}" .format( H/W ))
print("coverage = {}" .format( (W-unk)/W ))
