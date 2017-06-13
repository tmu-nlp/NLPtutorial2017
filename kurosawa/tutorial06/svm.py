import sys
from collections import defaultdict

class WEIGHT():
    def __init__(self):
        self.w = defaultdict(int)

    def update(self,count,y,c):
        for word,value in self.w.items():
            if abs(value) <= c:
                self.w[word] = 0
            else:
                self.w[word] -= sign(value) * c
        for word in count.keys():
            self.w[word] += count[word]*y

    def weight_get(self):
        return self.w

def sign(n):
    if n >= 0:
        return 1
    else:
        return -1

def class_check(count_class, weight_class):
    score = 0
    for word_class in count_class.keys():
        score += count_class[word_class]*weight_class[word_class]
    if score >= 0:
        return 1,score
    else:
        return -1,score

def word_count(words_c):
    count_c = defaultdict(int)
    for word in words_c:
        count_c[word] += 1
    return count_c

def fit(weight_,margin,c):
    with open('../../data/titles-en-train.labeled') as train:
        for line in train:
            label, words = line.lower().split('\t')
            label = int(label)
            words = words.split()
            count = word_count(words)
            class_now,score = class_check(count, weight_.weight_get())
            score = score * label
            if score <= margin:
                weight_.update(count,label,c)

if __name__ == '__main__':
    epoch = 20
    w = WEIGHT()
    margin = 50
    c = 0.0001
    for i in range(epoch):
        print(i)
        fit(w,margin,c)
    with open('../../data/titles-en-test.word') as test, open('my_answer.labeled','w') as answer:
        for line in test:
            words = line.lower().split()
            count = word_count(words)
            label,score = class_check(count, w.weight_get())
            answer.write('{}\t{}'.format(label,line))
