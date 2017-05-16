import sys
from collections import defaultdict
import math

def load_model(model_file):
    p_dict = defaultdict(float)
    with open(model_file, 'r') as f:
        for line in f:
            word, p = line.strip().split('\t')
            p_dict[word] = float(p)

    return p_dict

def test_unigram(test_file, p_dict):
    lambda_1 = 0.95
    lambda_unk = 1 - lambda_1
    V = 1000000
    W = 0
    H = 0
    unk = 0

    with open(test_file, 'r') as f:
        for line in f:
            word_list = line.strip().split()
            word_list.append("</s>")
            for word in word_list:
                W += 1
                P = lambda_unk / V
                if word in p_dict:
                    P += lambda_1 * p_dict[word]
                else:
                    unk += 1
                H += -math.log(P, 2)

    print('entropy = {}'.format(H/W))
    print('coverage = {}'.format(float(W-unk)/W))


if __name__ == "__main__":
    p_dict = load_model(sys.argv[1])
    test_unigram(sys.argv[2], p_dict)


