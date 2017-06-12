import sys
from collections import defaultdict
from train_perceptron_2 import class_check,word_count,word_count_bi

with open('model_bi_2.txt') as model:
    w_uni = defaultdict(int)
    w_bi = defaultdict(int)
    for line in model:
        flag, word, prob = line.strip().split(':=:')
        if flag == 'uni':
            w_uni[word] = int(prob)
        else:
            w_bi[word] = int(prob)

with open(sys.argv[1]) as test, open('my_answer_3.labeled','w') as answer:
    for line in test:
        words = line.lower().split()
        count = word_count(words)
        count_bi = word_count_bi(words)
        label = class_check(count,count_bi,w_uni,w_bi)
        answer.write('{}\t{}'.format(label,line))

