from collections import defaultdict
counts = defaultdict(lambda: 0)
c_counts = defaultdict(lambda: 0)

t_f = open('wiki-en-train.word','r')

for line in t_f:
    line = line.lower()
    words = line.split()
    words.insert(0,'<s>')
    words.append('</s>')
    for i in range(1,len(words)-1):
        counts[words[i-1] + ' ' + words[i]] += 1
        c_counts[words[i-1]] += 1
        counts[words[i]] += 1
        c_counts[""] += 1

m_f = open('model_file.txt','w')
for ngram, count in counts.items():
    words = ngram.split()
    del words[-1]
    context = ''.join(words)
    probability = counts[ngram]/c_counts[context]
    m_f.write('{}{}{}{}'.format(ngram,'\t',probability,'\n'))
