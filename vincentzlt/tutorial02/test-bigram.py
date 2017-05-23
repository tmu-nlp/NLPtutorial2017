
# coding: utf-8

# In[6]:

import sys
import math
from collections import defaultdict


# In[7]:

def read_model(f_name):
    n_gram_dict = defaultdict(lambda: 0)
    for line in open(f_name, "r", encoding="UTF-8"):
        line_split = line.split("\t")
        n_gram_dict[line_split[0].strip()] = float(line_split[1])

    return n_gram_dict


# In[8]:

def calc_lambda_2(n_gram_dict, bigram):
    uniq_word_after = 0
    count_word = 0
    first_word = bigram.split()[0]
    for key in n_gram_dict:
        if first_word == key.split()[0]:
            uniq_word_after += 1
    uniq_word_after -= 1
    return uniq_word_after / float(uniq_word_after + 34541)


# In[10]:

if __name__ == "__main__":
    lambda_1 = 0.95
    lambda_2=0.95
    V = 1000000.0
    W = 0.0
    H = 0.0
    sys.argv="self ./../../data/wiki-en-test.word ./bigram.model".split()
    n_gram_dict = read_model(sys.argv[2])
    for line in open(sys.argv[1], "r", encoding="UTF-8"):
        line_split = line.split()
        line_split.append("</s>")
        line_split.insert(0, "<s>")
        for i in range(1,len(line_split)-1):
            P1 = lambda_1 * n_gram_dict[line_split[i]] + (1 - lambda_1) / V
            bigram = " ".join(line_split[i:i+2])
            #lambda_2 = calc_lambda_2(n_gram_dict, bigram)
            P2 = lambda_2 * n_gram_dict[bigram] + (1 - lambda_2) * P1
            H += -math.log2(P2)
            W += 1
    print("entropy:\t", H / W)


# In[ ]:
