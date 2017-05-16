from collections import defaultdict
counts = defaultdict(int)
context_counts = defaultdict(int)

with open('../../data/wiki-en-train.word') as text:
    for line in text:
        words = line.split()
        words.append("</s>")
        words.insert(0,"<s>")
        for i in range(1,len(words)-1):
            counts[words[i-1]+" "+words[i]] += 1
            context_counts[words[i-1]] += 1
            counts[words[i]] += 1
            context_counts[""] += 1
   
with open('model_file.txt','w') as text:
    for ngram,count in counts.items():
        words = ngram.split()
        del words[-1]
        context = "".join(words)
        probability = counts[ngram]/context_counts[context]
        text.write(ngram+"\t"+str(probability)+"\n")







