#python train-unigram.py ../../data/wiki-en-train.word model.txt

from collections import defaultdict
import sys

def train_unigram(fin_path, model_path):
    word_count = defaultdict(lambda: 0)
    total_count = 0
    with open(fin_path, "r") as fin, open(model_path, "w") as fout:
        for line in fin:
            words = line.strip().split(" ")
            words.append("</s>")
            for word in words:
                word_count[word] += 1
                total_count += 1

        for word, freq in sorted(word_count.items(), key = lambda x: x[0]):
            fout.write("{} {}\n".format(word, float(freq) / total_count))

if __name__ == "__main__":
    train_unigram(sys.argv[1], sys.argv[2])
