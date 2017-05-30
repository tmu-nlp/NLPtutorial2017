from collections import defaultdict
import math

#best_edge = defaultdict(lambda: 0)
#best_edge = dict()
#best_score = defaultdict(lambda: 0)
unigram = dict()
w_count = defaultdict(int)
t_count = 0

lambda_1 = .95
lambda_unk = 1 - lambda_1
V = 10 ** 6
#"""
with open('../../data/wiki-ja-train.word','r') as t_f:
#m_f = open('wiki-ja-model.txt','w')

    for line in t_f:
        words = line.split()
        words.append('<\s>')
        for word in words:
            w_count[word] += 1
            t_count += 1
for word,count in w_count.items():
    probability = count/t_count
    unigram[word] = probability

print(unigram)
"""
    model = ('{}\t{}'.format(word,probability))
    m_f.write(model + '\n')



m_fr = open('wiki-ja-model.txt','r')

for line in m_fr:
    words = line.split()
    unigram[words[0]] = float(words[1])
"""
with open('../../data/wiki-ja-test.txt','r') as tt_f, open('my_answer.word','w') as a_f:
#a_f = open('my_answer.word','w')

    for line in tt_f:
        best_score = dict()
        best_edge = dict()
        best_edge[0] = None
        best_score[0] = 0
#    best_score = dict()
        for word_end in range(1, len(line) + 1):
            best_score[word_end] = 10 ** 10
            for word_begin in range(0, word_end):
                word = line[word_begin:word_end]
                if word in unigram or len(word) == 1:
                    prob = lambda_unk / V
                    if word in unigram:
                        prob += lambda_1 * unigram[word]
#                print(prob)
        #        prob = -1 * math.log2(prob)
        #        prob += best_score[word_begin]
                    my_score = best_score[word_begin] + (-1 * math.log2(prob))
        #        print(my_score)
                    if my_score < best_score[word_end]:
                        best_score[word_end] = my_score
                        best_edge[word_end] = (word_begin,word_end)
        #                print(best_edge)
        #                print(1)

        words = []
#    print(best_edge)
        next_edge = best_edge[len(best_edge) - 1]
        while next_edge != None:
            word = line[next_edge[0]:next_edge[1]]
            words.append(word)
            next_edge = best_edge[next_edge[0]]
        words.reverse()
        a_f.write(' '.join(words).strip(' '))
