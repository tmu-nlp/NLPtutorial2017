import sys
from collections import defaultdict
from train_perceptron import class_check,word_count

with open('model.txt') as model:
    w = defaultdict(int)
    for line in model:
        word, prob = line.split()
        w[word] = int(prob)

with open(sys.argv[1]) as test, open('my_answer.labeled','w') as answer:
    for line in test:
        words = line.split()
        count = word_count(words)
        label = class_check(count, w)
        answer.write('{}\t{}'.format(label,line))

