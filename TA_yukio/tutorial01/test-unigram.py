#python test-unigram.py ../../data/wiki-en-test.word model.txt 0.95 1000000

from collections import defaultdict
import sys
import math

def test_unigram(fin_path, model_path, lambda_1, vocabulary):
    word_token = 0
    unknown_token = 0
    H = 0

    probabilities = dict()
    with open(fin_path, "r") as fin, open(model_path, "r") as fm:
        for line in fm:
            word, prob = line.strip().split(" ")
            probabilities[word] = float(prob)

        for line in fin:
            words = line.strip().split(" ")
            words.append("</s>")
            for word in words:
                word_token += 1
                P = (1 - lambda_1) / vocabulary
                if word in probabilities:
                    P += lambda_1 * probabilities[word]
                else:
                    unknown_token += 1
                H -= math.log(P,2)

    print("Entropy = {}".format(H / word_token))
    print("Coverage = {}".format(float(word_token - unknown_token) / word_token))

if __name__ == "__main__":
    test_unigram(sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]))
