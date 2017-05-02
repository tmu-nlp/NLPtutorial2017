counts = dict()
with open('wiki-en-train.word') as f:
    for line in f:
        words = line.split()

        for w in words:
            if w in counts:
                counts[w] += 1
            else:
                counts[w] = 1

    for foo, bar in sorted(counts.items()):
        print('{0} -> {1}' .format(foo, bar))
