from collections import defaultdict
import sys

def main(input_file):
    emit = defaultdict(int)
    transition = defaultdict(int)
    context = defaultdict(int)
    with open(input_file, 'r') as f:
        for line in f:
            prev = '<s>'
            context[prev] += 1
            wordtags = line.strip().split()
            for wordtag in wordtags:
                word, tag = wordtag.split("_")
                transition[prev+' '+tag] += 1
                context[tag] += 1
                emit[tag+' '+word] += 1
                prev = tag
            transition[prev+' </s>'] += 1
        for k, v in transition.items():
            prev, tag = k.split(" ")
            print("T {} {}".format(k, v/context[prev]))
        for k, v in emit.items():
            prev, tag = k.split(" ")
            print("E {} {}".format(k, v/context[prev]))

if __name__ == "__main__":
    main(sys.argv[1])

