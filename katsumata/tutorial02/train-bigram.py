import sys
from collections import defaultdict
counts = defaultdict(int)
context_counts = defaultdict(int)
with open(sys.argv[1], 'r') as training_file:
    for line in training_file:
        line = line.lower()
        words = line.split()
        words.append('</s>')
        words.insert(0, '<s>')
        for i in range(1,len(words)-1):
            temp_str1 = ''
            temp_str1 = ' '.join(words[i-1:i+1])
            unigram_word = ''
            unigram_word += words[i]
            counts[temp_str1] += 1 #分子
            context_counts[words[i-1]] += 1 #分母
            counts[unigram_word] += 1 #分子
            context_counts[''] += 1 #分母
with open('model_file.txt', 'w') as m_f:
    for ngram, count in counts.items():
        words = ngram.split()
        words.pop()
        context = ''.join(words)
        probability = float(counts[ngram]/context_counts[context])
        s = ('{}:{}'.format(ngram, probability))
        m_f.write(s + '\n')
