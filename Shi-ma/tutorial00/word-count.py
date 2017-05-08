import collections

with open('../../data/wiki-en-train.word', 'r') as data:
    word_count = collections.defaultdict(lambda :0)
    for line in data:
        for word in line.split():
            word_count[word] += 1
for i in sorted(word_count.items(), key=lambda x: x[1], reverse = 1):
    print(*i)
print(len(word_count))
