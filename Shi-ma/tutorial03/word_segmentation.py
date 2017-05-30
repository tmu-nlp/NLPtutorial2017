import collections
import math
import unicodedata
import re

def train(data):
    total_count = 0
    word_count = collections.defaultdict(lambda :0)
    word_probabilities = collections.defaultdict(lambda :0)
    for line in data:
        for unigram in line.split():
            total_count += 1
            word_count[unigram] += 1
    for word in word_count.keys():
        word_probabilities[word] = word_count[word]/total_count
    return word_probabilities

def word_segmentation(data, word_probabilities, data_out):
    rate_unk = 0.05; N = 1000000
    pattern = re.compile(r'^[a-zA-Z]+$')
    for line in data:
        best_edge = [None for i in range(len(line))]
        best_score = [10**10 for i in range(len(line))]
        best_score[0] = 0
        for word_end in range(1, len(line)):
            for word_begin in range(0, word_end):
                word = line[word_begin:word_end]
                word_alpha = pattern.match(unicodedata.normalize('NFKC', word))
                if word in word_probabilities.keys() or len(word) == 1 or word_alpha:
                    if word in word_probabilities.keys():
                        prob = (1 - rate_unk)*word_probabilities[word] + rate_unk/N
                    else:
                        prob = rate_unk/N
                    if word_alpha:
                        prob = 0.05*len(word)
                    my_score = best_score[word_begin] + - math.log2(prob)
                    if my_score < best_score[word_end]:
                        best_score[word_end] = my_score
                        best_edge[word_end] = (word_begin, word_end)
        words = []
        next_edge = best_edge[-1]
        while next_edge is not None:
            word = line[next_edge[0]:next_edge[1]]
            words.append(word)
            next_edge = best_edge[next_edge[0]]
        words.reverse()
        data_out.write(' '.join(words) + '\n')

if __name__ == '__main__':
    with open('../../data/wiki-ja-train.word', 'r') as data_train:
        word_probabilities = train(data_train)
    with open('../../data/wiki-ja-test.txt', 'r') as data_test:
        with open('my_answer.word', 'w') as data_out:
            word_segmentation(data_test, word_probabilities, data_out)
