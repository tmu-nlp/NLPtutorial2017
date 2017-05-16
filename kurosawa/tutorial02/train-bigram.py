import sys
from collections import defaultdict

counts = defaultdict(int)
context_counts = defaultdict(int)

with open(sys.argv[1], "r") as train:
    for words in train:
        words = words.lower().split()
        words.insert(0,"<s>")
        words.append("</s>")
        for i in range(1,len(words)):
            counts[str(words[i-1]+' '+words[i])] += 1
            context_counts[words[i-1]] += 1
            counts[words[i]] += 1
            context_counts[""] += 1
#    print(context_counts["Moreover"])
            
with open("model.txt","w") as model:
    for ngram, count in counts.items():
#        print(ngram)
        words = ngram.split()
        words.pop()
        context = ''.join(words)
        probability = counts[ngram]/context_counts[context]
        model.write('{}\t{}\n'.format(ngram,probability))
