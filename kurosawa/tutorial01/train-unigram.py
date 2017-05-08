import sys
from collections import defaultdict

counts = defaultdict(int)
total = 0

with open(sys.argv[1], "r") as lines:
    for line in lines:
        line = line.lower()
        words = line.split()
        words.append('</s>')
#        print(words)
        for word in words:
            counts[word] += 1
            total += 1
#print(counts['</s>'])
with open('model.txt', "w") as model:
    for word, count in counts.items():
        probability = count/total
        model.write('{} {}\n'.format(word, probability))
