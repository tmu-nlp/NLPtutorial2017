import sys
import math
from collections import defaultdict

model = defaultdict(int)
with open('model.txt') as model_f:
    for line in model_f:
        line = line.split()
        model[line[0]] = float(line[1])

N = 1000000
lambda_1 = 0.95
lambda_unk = 1-lambda_1
with open(sys.argv[1]) as input_file:
    with open('my_answer.word','w') as write_file:
        for line in input_file:
            best_score = defaultdict(lambda:10**10)
            best_edge = defaultdict(tuple)
            best_edge[0] = 'NULL'
            best_score[0] = 0
            for end in range(1,len(line)+1):
                for begin in range(len(line)):
                    word = line[begin:end]
                    if (word in model) or (len(word) == 1):
                        prob = lambda_1 * model[word] + lambda_unk / N
                        prob = -1 * math.log2(prob)
                        prob += best_score[begin]
                        if prob < best_score[end]: #イコールどうする？
                            best_score[end] = prob
                            best_edge[end] = (begin, end)
                            print('best_score[{}] : {}'.format(end,best_score[end]))
                            print('best_edge[{}] : {}'.format(end,best_edge[end]))
#前向き処理終了？
            word_split = []
            next_edge = best_edge[len(line)]
            while next_edge != 'NULL':
                begin_split, end_split = next_edge
                word_split.append(line[begin_split:end_split])
                next_edge = best_edge[begin_split]
            word_split.reverse()
            words = ' '.join(word_split).strip(' ')
            write_file.write('{}'.format(words))
