import sys
from collections import defaultdict
word_count = defaultdict(lambda: 0)
total_count = 0
with open(sys.argv[1], 'r') as f:
    for row in f:
        words = row.split()
        words.append('</s>')
        for word in words:
            word_count[word.lower()] += 1
            #word_count[word] += 1
            total_count += 1
with open('model_file.txt', 'w') as f_w:
    for word, count in (word_count.items()):
        probability = count/total_count
        str_temp = ('{} : {}'.format(word, probability) )
        f_w.write(str_temp + '\n')
