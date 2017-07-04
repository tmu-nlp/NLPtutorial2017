
# coding: utf-8

# In[56]:


import numpy as np
import math
from collections import defaultdict
from pprint import pprint


# In[42]:


def sampleone(probs):
    z=np.sum(probs)
    remaining=np.random.rand()*z
    for i in range(len(probs)):
        remaining-=probs[i]
        if remaining<=0:
            return i


# In[43]:


test_file="../../test/07-train.txt"


# In[44]:


def gen_ids(f_name):
    ids={}
    for line in open(f_name,"r",encoding="utf-8"):
        for w in line.split():
            if w not in ids:
                ids[w]=len(ids)
    return ids


# In[45]:


gen_ids(test_file)


# In[58]:


x_corpus,y_corpus=[],[]
x_counts,y_counts=defaultdict(int),defaultdict(int)
num_topics=2
for line in open(test_file,"r",encoding="utf-8"):
    docid=len(x_corpus)
    words=line.split()
    topics=[np.random.randint(num_topics) for word in words]
    for word,topic in zip(words,topics):
        addcounts(word,topic,docid,1)
    x_corpus.append(words)
    y_corpus.append(topics)
    


# In[47]:


def addcounts(word,topic,docid,amount):
    x_counts[topic]+=amount
    x_counts[(word,topic)]+=amount
    
    y_counts[docid]+=amount
    y_counts[(topic,docid)]+=amount


# In[48]:


x_corpus
y_corpus
x_counts
y_counts


# In[59]:


for iters in range(1000):
    II = 0
    for i in range(len(x_corpus)):
        for j in range(len(x_corpus[i])):
            x = x_corpus[i][j]
            y = y_corpus[i][j]
            addcounts(x, y, i, -1)
            probs = []
            for k in range(num_topics):
                probs.append((x_counts[(x, k)] / y_counts[k]) *
                             (y_counts[k] / len(x_corpus[i])))
            new_y = sampleone(probs)
            try:
                II += np.log(probs[new_y])
            except:
                pass
            addcounts(x, new_y, i, 1)
            y_corpus[i][j] = new_y
            pprint(x_corpus)
            pprint(y_corpus)
            pprint(x_counts)
            pprint(y_counts)
            input()
    print(II)
print(len(set([w for w in line for line in x_corpus])))
print(len(set([t for t in line for line in y_corpus])))


# In[ ]:




