#python train-bigram.py ../../data/wiki-en-train.word model.txt

from collections import defaultdict
import sys

def train_bigram(fin_path, model_path):
    counts = defaultdict(lambda: 0)
    context_counts = defaultdict(lambda: 0)
    with open(fin_path, "r") as fin, open(model_path, "w") as fout:
        for line in fin:
            words = line.strip().split(" ")
            words.append("</s>")
            words.insert(0, "<s>")
            for i in range(1, len(words)):
                counts["{} {}".format(words[i - 1], words[i])] += 1
                context_counts[words[i - 1]] += 1
                counts[words[i]] += 1
                context_counts[""] += 1

        for ngram, count in sorted(counts.items(), key = lambda x: x[0]):
            words = ngram.split()
            del words[-1]
            context = "".join(words)
            fout.write("{}\t{}\n".format(ngram, float(counts[ngram]) / context_counts[context]))

if __name__ == "__main__":
    train_bigram(sys.argv[1], sys.argv[2])
