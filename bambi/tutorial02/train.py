
# coding: utf-8

# In[5]:

from collections import defaultdict
import math

counts = defaultdict(int)
context_counts = defaultdict(int)

#../test/02-train-input.txt
path = "../data/wiki-en-train.word"
with open(path) as training_file, open("model_file.txt","w") as output:
    for line in training_file.readlines():
        words = line.split()
        words.insert(0,"<s>")
        words.append("</s>")
        for i in range(1,len(words)-1):
            context_counts[words[i-1]] += 1.0
            counts["{} {}".format(words[i-1],words[i])] += 1.0
            counts[words[i]] +=1.0
            context_counts[""] += 1.0
        #print(counts)
    for ngram, count in counts.items():
        words = ngram.split()
        words.pop()
        context = "".join(words)
        probability = counts[ngram]/context_counts[context]
        output.write("{}\t{}\n".format(ngram, probability))
        #print(ngram, probability)


# In[ ]:




# In[ ]:



