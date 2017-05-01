import sys
from collections import defaultdict
word_count = defaultdict(lambda: 0)
with open(sys.argv[1], 'r') as f:
    for row in f:
        row = row.lower()
        words = row.split()
        for i in range(len(words)):
            word_count[words[i]] += 1

    for word, count in sorted(word_count.items(),reverse=True,key=lambda x:x[1]):
        print('%s : %d' %(word, count))
