from collections import defaultdict

counts = defaultdict(lambda: 0)
total_count = 0
with open("wiki-en-train.word", "r") as f1:
        for line in f1:
            line = line.lower()
            words = line.split()
            words.append("<\s>")
            for w in words:
                counts[w] += 1
                total_count += 1


with open("modelfile.txt", "w") as f2:
    for word2 in sorted(counts):
        probability = float(counts[word2]) / float(total_count)
        f2.write("{}    {}\n" .format(word2, probability))
