
# coding: utf-8

# In[13]:


from pprint import pprint
from collections import defaultdict
import math


# In[48]:


f_grammar="../../data/wiki-en-test.grammar"
f_input="../../data/wiki-en-short.tok"


# In[50]:


nonterms=[]
preterms=defaultdict(list)
for line in open(f_grammar,"r",encoding="utf-8"):
    num_tab=line.count("\t")
    if num_tab==2:
        lhs,rhs,prob=line.split("\t")
        prob=float(prob)
        if " " in rhs:
            rhs1,rhs2=rhs.split(" ")
            nonterms.append((lhs,rhs1,rhs2,prob))
        else:
            preterms[rhs].append((lhs,prob))


# In[51]:


def print_tree(node):
    if node in best_edges:
        return "({node_tag} {best_node0} {best_node1})".format(
            node_tag=node[0],
            best_node0=print_tree(best_edges[node][0]),
            best_node1=print_tree(best_edges[node][1]))
    else:
        return "({node_tag} {word} )".format(
            node_tag=node[0],
            word=line[node[1]])
        

# In[52]:


best_scores = defaultdict(lambda: -10000.0)
best_edges = {}
for line in open(f_input, "r", encoding="utf-8"):
    line = line.split()
    for idx, word in enumerate(line):
        l_idx = idx
        r_idx = idx + 1
        for lhs, prob in preterms[word]:
            best_scores[(lhs, l_idx, r_idx)] = -math.log(prob)
    for j in range(2, len(line)+1):
        for i in reversed(range(j - 1)):
            for k in range(i + 1, j):
                for sym, lsym, rsym, logprob in nonterms:
                    logprob = -math.log(logprob)
                    if best_scores[(lsym, i, k)] > -10000.0 and best_scores[(
                            rsym, k, j)] > -10000.0:
                        my_lp = best_scores[(lsym, i, k)] + best_scores[(
                            rsym, k, j)] + logprob
                        if my_lp > best_scores[(sym, i, j)]:
                            best_scores[(sym, i, j)] = my_lp
                            best_edges[(sym, i, j)] = ((lsym, i, k), (rsym, k,
                                                                      j))
    
#     best_scores
#     best_edges
    print(print_tree(("S",0,len(line))))


# In[45]:



