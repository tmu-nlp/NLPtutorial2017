import sys
import math
from collections import defaultdict

lambda_1 = 0.95
lambda_unk = 1 - lambda_1
V = 1000000
W = 0
H = 0
unk = 0

probabilities = defaultdict(int)
with open('model.txt', "r") as model:
    for line in model:
        line = line.split()
        probabilities[line[0]] = float(line[1])
with open(sys.argv[1], "r") as test:
   for line in test:
       line = line.lower()
       words = line.split()
       words.append('</s>')
#       print(words)
       for w in words:
           W += 1
           P = lambda_unk / V
           if w in probabilities:
               P += lambda_1 * probabilities[w]
           else:
               unk += 1
#               print(w,unk)
           H += -1 * math.log(P, 2)
#   print(H)
#   print(W)
#   print(unk)
   print ("entropy = "+ str(H / W))
   print ("coverage = "+ str((W-unk) / W))

