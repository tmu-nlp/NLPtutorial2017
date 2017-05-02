from collections import defaultdict

# SRC = '01-train-input.txt'
SRC = '../../data/wiki-en-train.word'

# GEN_MODEL_FILE = "sample.model"
GEN_MODEL_FILE = "wiki.model"

count = defaultdict(lambda: 0)
total_count = 0

with open(SRC) as f:
    for line in f:
        l = line.strip('\n').lower().split(' ')
        l.append('</s>')
        for w in l:
            count[w] += 1
            total_count += 1

with open(GEN_MODEL_FILE, 'w') as f:
    for k, v in count.items():
        probability = v/total_count
        f.write(k + " " + str(probability) + '\n')
