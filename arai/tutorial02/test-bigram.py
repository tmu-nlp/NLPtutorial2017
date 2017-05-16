from collections import defaultdict
import math
lambda1 = 0.95
lambda2 = 0.95
V = 10**6
W = 0
H = 0

with open('model_file.txt') as text:
    probs = defaultdict(int)
    for line in text:
        words =line.split("\t")
        probs[words[0]] = words[1]

with open('../../data/wiki-en-test.word') as text:
    for line in text:
        words = line.split()
        words.append("</s>")
        words.insert(0,"<s>")
        for i in range(1,len(words)-1):
            P1 = lambda1*float(probs[words[i]])+(1-lambda1)/V
            P2 = lambda2*float(probs[words[i-1]+" "+words[i]])+(1-lambda2)*P1
            H += -math.log2(P2)
            W += 1
    print("entropy =" +str(H/W))
