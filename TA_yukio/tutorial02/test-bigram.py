#python test-bigram.py ../../data/wiki-en-test.word model.txt 0.95 1000000

from collections import defaultdict
import sys
import math

def test_bigram(fin_path, model_path, lambda_1, vocabulary):
    word_token = 0
    H = 0

    probabilities = defaultdict(lambda: 0.0)
    with open(fin_path, "r") as fin, open(model_path, "r") as fm:
        for line in fm:
            ngram, prob = line.strip().split("\t")
            probabilities[word] = float(prob)

        for line in fin:
            words = line.strip().split(" ")
            words.append("</s>")
            words.insert(0, "<s>")
            for i in range(1, len(words)):
          
                P1 = lambda_1 * probabilities[words[i]] + (1 - lambda_1) / vocabulary
                P2 = lambda_2 * probabilities["{} {}".format(words[i - 1], words[i])] + (1 - lambda_2) * P1
                H -= math.log(P2,2)
                word_token += 1

    print("Entropy = {}".format(H / word_token))

if __name__ == "__main__":
    test_bigram(sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]))

#
lambda_1 = float(sys.argv[1])
lambda_2 = float(sys.argv[2])
