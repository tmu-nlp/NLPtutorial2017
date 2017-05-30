#python test-viterbi.py ../../data/wiki-ja-train.word model.txt ../../data/wiki-ja-test.txt my_answer.word 0.95 1000000
#../../script/gradews.pl ../../data/wiki-ja-test.word my_answer.word

from collections import defaultdict
import sys
import math

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

def test_viterbi(fin_path, model_path, fout_path, lambda_1, vocabulary):
    unigram_probabilities = defaultdict(lambda: 0.0)
    with open(fin_path, "r") as fin, open(model_path, "r") as fm, open(fout_path, "w") as fout:
        for line in fm:
            words = line.split(" ")
            unigram_probabilities[words[0]] = float(words[1])
        for line in fin:
            best_edge = dict()
            best_score = dict()
            line = line.strip()
            best_edge[0] = "NULL"
            best_score[0] = 0
            for word_end in range(1, len(line) + 1):
                best_score[word_end] = 10 ** 10
                for word_begin in range(0, word_end):
                    word = line[word_begin:word_end]
                    if word in unigram_probabilities or len(word) == 1:
                        prob = lambda_1 * unigram_probabilities[word] + (1 - lambda_1) / vocabulary
                        my_score = best_score[word_begin] - math.log(prob, 2)
                        if my_score < best_score[word_end]:
                            best_score[word_end] = my_score
                            best_edge[word_end] = (word_begin, word_end)
            words = list()
            next_edge = best_edge[len(best_edge) - 1]
            while next_edge != "NULL":
                word = line[next_edge[0]:next_edge[1]]
                words.append(word)
                next_edge = best_edge[next_edge[0]]
            words.reverse()
            fout.write(" ".join(words) + "\n")

if __name__ == "__main__":
    train_unigram(sys.argv[1], sys.argv[2])
    test_viterbi(sys.argv[3], sys.argv[2], sys.argv[4], float(sys.argv[5]), int(sys.argv[6]))
