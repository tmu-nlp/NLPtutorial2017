import math
from collections import defaultdict
import sys

def load_model(model_file):
    p_dict = defaultdict(float)
    with open(model_file, 'r') as f:
        for line in f:
            word, p = line.strip().split('\t')
            p_dict[word] = float(p)

    return p_dict

def main(input_file):
    # for unigram model
    lambda_1 = 0.95
    lambda_unk = 1 - lambda_1
    V = 1000000

    with open(input_file, 'r') as f:
        for line in f:
            # forward step
            best_edge = defaultdict(str)
            best_score = defaultdict(int)
            best_edge[0] = 'NULL'
            best_score[0] = 0
            line = line.strip()  # remove new line
            for end_i in range(1,len(line)+1):
                best_score[end_i] = 10**10
                for begin_i in range(0, end_i):  # end_i - 1 + 1
                    word = line[begin_i:end_i]
                    if word in p_dict or len(word) == 1:
                        prob = lambda_1 * p_dict[word] + lambda_unk / V
                        my_score = best_score[begin_i] - math.log(prob, 2)
                        if my_score < best_score[end_i]:
                            best_score[end_i] = my_score
                            best_edge[end_i] = (begin_i, end_i)
            
            # back step
            word_list = list()
            next_edge = best_edge[len(best_edge)-1]  # next_edge=(n, n_1)
            while next_edge != 'NULL':
                word = line[next_edge[0]:next_edge[1]]
                word_list.append(word)
                next_edge = best_edge[next_edge[0]]  # next_edge=(n_-1, n)
            word_list.reverse()
            print(' '.join(word_list))

if __name__ == "__main__":
        p_dict = load_model(sys.argv[1])  # ja-unigram-prob.txt 
        main(sys.argv[2])  # data/wiki-ja-test.txt
