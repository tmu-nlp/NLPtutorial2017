import sys
from collections import defaultdict
import math

def load_model(model_file):
    p_dict = defaultdict(float)
    with open(model_file, 'r') as f:
        for line in f:
            ngram, p = line.strip().split('\t')
            p_dict[ngram] = float(p)

    return p_dict

def test_bigram(test_file, p_dict):
    lambda_1 = 0.95
    lambda_2 = 0.95 
    lambda_unk1 = 1 - lambda_1
    lambda_unk2 = 1 - lambda_2
    V = 1000000
    W = 0
    H = 0

    with open(test_file, 'r') as f:
        for line in f:
            word_list = line.strip().split()
            word_list.append("</s>")
            word_list.insert(0, "<s>")
            for i, word in enumerate(word_list[:-1]):
                P1 = lambda_1 * p_dict[word] + lambda_unk1 / V
                P2 = lambda_2 * p_dict[word+' '+word_list[i+1]] + lambda_unk2 * P1
                H += -math.log(P2, 2)
                W += 1

    print('entropy = {}'.format(H/W))


if __name__ == "__main__":
    p_dict = load_model(sys.argv[1])
    test_bigram(sys.argv[2], p_dict)

