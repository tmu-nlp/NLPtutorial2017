#python train-hmm.py ../../data/wiki-en-train.norm_pos model.txt

from collections import defaultdict
import sys

def train_hmm(fin_path, model_path):
    emit = defaultdict(lambda: 0)
    transition = defaultdict(lambda: 0)
    context = defaultdict(lambda: 0)

    with open(fin_path, "r") as fin, open(model_path, "w") as fout:
        for line in fin:
            line = line.strip("\n")
            previous = "<s>"
            context[previous] += 1
            wordtags = line.split(" ")
            for wordtag in wordtags:
                word, tag = wordtag.split("_")
                transition["{} {}".format(previous, tag)] += 1
                context[tag] += 1
                emit["{} {}".format(tag, word)] += 1
                previous = tag
            transition["{} </s>".format(previous)] += 1

        for key, value in sorted(transition.items()):
            previous, word = key.split(" ")
            fout.write("T {} {}\n".format(key, value / context[previous]))

        for key, value in sorted(emit.items()):
            tag, word = key.split(" ")
            fout.write("E {} {}\n".format(key, value / context[tag]))

if __name__ == "__main__":
    train_hmm(sys.argv[1], sys.argv[2])
