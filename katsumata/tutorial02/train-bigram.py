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
        #ここまではオッケー
        #print (words)
        for i in range(1,len(words)-1):
            """
            #temp_str1 = ''
            temp_str1 = tuple(words[i-1:i+1])
            #print (words[i-1:i+1])
            #temp_str1 += '{} {}'.format(words[i-1:i+1])
            #print (temp_str1)
            unigram_word = (words[i],)
            #print (unigram_word)
            """
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
        """
        temp_str2 = ''
        #word = list(ngram)
        #print (word)
        #word.pop()
        temp_str2 += ngram[0]
        if context_counts[temp_str2] == 0:
            print ('ngram: tempstr2 is {}'.format(temp_str2))
            break
        """
        words = ngram.split()
        words.pop()
        context = ''.join(words)
        probability = counts[ngram]/context_counts[context]
        #print ('{} : {}'.format(ngram, probability))
        s = ('{}:{}'.format(ngram, probability))
        m_f.write(s + '\n')
"""
for key, value in counts.items():
    print ('{} : {}'.format(key, value))
"""    
