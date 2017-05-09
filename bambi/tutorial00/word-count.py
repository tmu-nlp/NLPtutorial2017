import sys
from collections import defaultdict
input_file = open(sys.argv[1],"r")

word_counts = defaultdict(lambda:0)
for line in input_file:
    line = line.strip()
    if len(line) != 0:
        words = line.split(" ")
        for word in words:
            if word not in word_counts:
                word_counts[word] = 1
            else:
                word_counts[word] += 1

for key, value in sorted(word_counts.items(),key = lambda x:x[1], reverse = True)[:10]:
    print("{}: {}".format(key, value))
