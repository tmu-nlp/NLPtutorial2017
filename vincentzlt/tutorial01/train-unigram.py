
# coding: utf-8

# In[1]:

import sys, math
from collections import defaultdict


# In[24]:

def gen_corpus_list(f_name):
    with open(f_name, "r", encoding="UTF-8") as f:
        corpus_list = [line.split() for line in f.readlines()]
    for line in corpus_list:
        line.append("<s/>")
        #line.insert(0, "<s>")
    return corpus_list


# In[25]:

def gen_unigram_dict(corpus_list):
    unigram_dict=defaultdict(lambda :0)
    
    for line in corpus_list:
        for w in line:
            unigram_dict[w]+=1
    return unigram_dict


# In[26]:

def calc_sum_token(corpus_list):
    return len([w for line in corpus_list for w in line])


# In[29]:

def calc_unigram_model(corpus_list):

    unigram_dict = gen_unigram_dict(corpus_list)
    sum_token = calc_sum_token(corpus_list)
    
    lambda_unigram=0.95
    vocab_size=1000000
    
    for w in unigram_dict:
        P_ml=unigram_dict[w]/float(sum_token)

        unigram_P=(lambda_unigram*P_ml+(1-lambda_unigram)/vocab_size)
        print(w,"\t", P_ml)
    return 


# In[30]:

if __name__=="__main__":
    #sys.argv="self ../../test/01-train-input.txt".split()
    corpus_list=gen_corpus_list(sys.argv[1])
    calc_unigram_model(corpus_list)


# In[ ]:



