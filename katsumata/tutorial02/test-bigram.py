from collections import defaultdict
import sys
import math
probs = defaultdict(lambda : .0) 
lambda1 = .85
lambda2 = .15
V = 10 ** 6
W = 0
H = 0

with open('model_file.txt', 'r') as m_f:
    for row in m_f:
        words = row.strip().split(':')
        #print (words)
        probs[words[0]] = words[1]
        #print (words[1])
#print (probs)
for key, value in probs.items():
    print ('{} : {}'.format(key, value))

with open(sys.argv[1], 'r') as t_f:
    for line in t_f:
        line = line.lower()
        words = line.split()
        words.append('</s>')
        words.insert(0, '<s>')
        for i in range(1, len(words)-1):
            temp_str1 = ' '.join(words[i-1:i+1])
            print (temp_str1)
            P1 = lambda1 * float(probs[words[i]]) + (1 - lambda1) / V
            P2 = lambda2 * float(probs[temp_str1]) + (1 - lambda2) * P1
            H += -1 * math.log2(P2)
            W += 1
print ('entropy = {}'.format(H/W)) 
