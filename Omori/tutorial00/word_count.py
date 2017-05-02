import sys
from collections import defaultdict

input_f = open(sys.argv[1],'r')
word_dict = defaultdict(int)

for line in input_f:
    words = line.strip().split()
    for word in words:
        word = word.lower()
        word_dict[word] += 1       

print(len(word_dict))

for key, value in sorted(word_dict.items(), key=lambda x: -x[1])[:10]:
    print('{}: {}'.format(key, value))


