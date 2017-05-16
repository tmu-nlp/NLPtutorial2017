from collections import defaultdict

counts_u = defaultdict(lambda: 0)
counts_bi = defaultdict(lambda: 0)
total_count = 0
bigram = list()
with open("../../data/wiki-en-train.word", "r") as f1:
        for line in f1:
            line = line.lower()
            words = line.split()
            words.insert(0,'<s>')
            words.append("<\s>")
            for i in range(1, len(words) - 1):
                counts_u[words[i-1]] += 1
                total_count += 1
                counts_bi[words[i-1], words[i]] += 1

with open("modelfile.txt", "w") as f2:
    for word1 in counts_bi.keys():
        probability = float(counts_bi[word1] / counts_u[word1[0]])
        f2.write("{}    {}\n" .format(" ".join(word1), probability))
