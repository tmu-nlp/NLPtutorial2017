
# coding: utf-8

# In[94]:


import numpy as np
import math
from collections import defaultdict
from pprint import pprint


# In[95]:


def sampleone(probs):
    z=np.sum(probs)
    remaining=np.random.rand()*z
    for i in range(len(probs)):
        remaining-=probs[i]
        if remaining<=0:
            return i


# In[96]:


test_file="../../data/wiki-en-documents.word"


# In[97]:


def gen_ids(f_name):
    ids={}
    for line in open(f_name,"r",encoding="utf-8"):
        for w in line.split():
            if w not in ids:
                ids[w]=len(ids)
    return ids


# In[101]:


x_corpus,y_corpus=[],[]
x_counts,y_counts=defaultdict(int),defaultdict(int)
num_topics=20
smoothing_a=0.001
smoothing_b=0.001


# In[102]:


for line in open(test_file,"r",encoding="utf-8"):
    docid=len(x_corpus)
    words=line.split()
    topics=[np.random.randint(num_topics) for word in words]
    for word,topic in zip(words,topics):
        addcounts(word,topic,docid,1)
    x_corpus.append(words)
    y_corpus.append(topics)
    
Nx=len(set(w for line in x_corpus for w in line ))
Ny=len(set(t for line in y_corpus for t in line ))


# In[103]:


def addcounts(word,topic,docid,amount):
    x_counts[topic]+=amount
    x_counts[(word,topic)]+=amount
    
    y_counts[docid]+=amount
    y_counts[(topic,docid)]+=amount


# In[ ]:


for iters in range(10):
    II = 0
    for i in range(len(x_corpus)):
        for j in range(len(x_corpus[i])):
            x = x_corpus[i][j]
            y = y_corpus[i][j]
            addcounts(x, y, i, -1)
            probs = []
            for k in range(num_topics):
                probs.append(((x_counts[(x, k)]+ smoothing_a) / (y_counts[k]+smoothing_a*Nx)) *
                             ((y_counts[(k,i)]+smoothing_b) / (len(x_corpus[i])+smoothing_b*Ny)))
            #print(probs)
            new_y = sampleone(probs)
            #print(new_y)
            try:
                II += np.log(probs[new_y])
            except:
                pass
            addcounts(x, new_y, i, 1)
            y_corpus[i][j] = new_y
#             pprint(x_corpus)
#             pprint(y_corpus)
#             pprint(x_counts)
#             pprint(y_counts)
            
    print(II)
print(len(set(w  for line in x_corpus for w in line)))
print(len(set(t for line in y_corpus for t in line )))


# In[75]:


x_counts
y_counts

