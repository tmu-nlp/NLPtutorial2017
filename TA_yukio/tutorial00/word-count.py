#python word-count.py ../data/wiki-en-train.word

from collections import defaultdict
import sys

def word_count(fin_path):
    word_count = defaultdict(lambda: 0)
    with open(fin_path, "r") as fin:
        for line in fin:
            for word in line.strip().split(" "):
                word_count[word] += 1

        print("Word Type: {}".format(len(word_count)))
        for word, freq in sorted(word_count.items(), key = lambda x: x[0]):
            print("{} {}".format(word, freq))

if __name__ == "__main__":
    word_count(sys.argv[1])
