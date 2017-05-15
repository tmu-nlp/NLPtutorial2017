
# coding: utf-8

# In[27]:

import math
from collections import defaultdict

lamb1 = 0.95
lamb2 = 0.1
V = 1000000
W = 0
H = 0
probs = defaultdict(int)
lines = open("model_file.txt", "r").readlines()
for line in lines:
    words = line.split("\t")
    probs[words[0]] = float(words[1])
    
with open("../data/wiki-en-test.word") as test_file:
    for line in test_file.readlines():
        words = line.split()
        words.insert(0,"<s>")
        words.append("</s>")
        for i in range(1,len(words) -1):
            P1 = lamb1 * probs[words[i]] + (1-lamb1)/V
            P2 = lamb2 * probs["{} {}".format(words[i-1],words[i])] + (1-lamb2) * P1
            H -= math.log(P2,2)
            W += 1
    print("entropy = {}".format(H/W))


# In[ ]:




# In[ ]:



