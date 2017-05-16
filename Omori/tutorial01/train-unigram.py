import sys
from collections import defaultdict


def train_unigram(input_file, output_file):
    word_count = defaultdict(int)
    total = 0
    with open(input_file, 'r') as f:
        for line in f:
            word_list = line.strip().split()
            word_list.append("</s>")
            for word in word_list:
                word_count[word] += 1
                total += 1

    with open(output_file, 'w') as f:
        for word, count in sorted(word_count.items(), key=lambda x: -x[1]):
            p = count / total
            f.write(word+'\t'+str(p)+'\n')

if __name__ == "__main__":
    train_unigram(sys.argv[1], sys.argv[2])

