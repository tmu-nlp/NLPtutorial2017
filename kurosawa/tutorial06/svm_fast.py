import sys
from collections import defaultdict

class LAST():
    def __init__(self):
        self.last = defaultdict(int)

    def update(self,word,iter_):
        self.last[word] = iter_

    def get_last(self,word):
        return self.last[word]

class WEIGHT():
    def __init__(self):
        self.w = defaultdict(int)

    def update(self,count,y):
        for word in count.keys():
            self.w[word] += count[word]*y

    def weight_get_update(self,word,c,iter_,last):
        last_iter = last.get_last(word)
        if iter_ != last_iter:
            c_size = c * (iter_ - last_iter)
            if abs(self.w[word]) <= c_size:
                self.w[word] = 0
            else:
                self.w[word] -= sign(self.w[word]) * c_size
            last.update(word,iter_)
        return self.w[word]

    def weight_update(self,c,iter_,last):
        for word in self.w.keys():
            last_iter = last.get_last(word)
            if iter_ != last_iter:
                c_size = c * (iter_ - last_iter)
                if abs(self.w[word]) <= c_size:
                    self.w[word] = 0
                else:
                    self.w[word] -= sign(self.w[word]) * c_size

    def weight_get(self,word):
        return self.w[word]

def sign(n):
    if n >= 0:
        return 1
    else:
        return -1

def class_check(count_class, weight_class,mode,c=0,iter_=0,last=0):
    score = 0
    for word_class in count_class.keys():
        if mode == 1:
            score += count_class[word_class]*weight_class.weight_get_update(word_class,c,iter_,last)
        else:
            score += count_class[word_class]*weight_class.weight_get(word_class)
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
        last = LAST()
        for iter_, line in enumerate(train):
            label, words = line.split('\t')
            label = int(label)
            words = words.split()
            count = word_count(words)
            class_now,score = class_check(count, weight_,1,c,iter_+1,last)
            score = score * label
            if score <= margin:
                weight_.update(count,label)
        weight_.weight_update(c,iter_+1,last)

if __name__ == '__main__':
    epoch = 20
    w = WEIGHT()
    margin = 80
    c = 0.0001
    for i in range(epoch):
        print(i)
        fit(w,margin,c)
    with open('../../data/titles-en-test.word') as test, open('my_answer_fast.labeled','w') as answer:
        for line in test:
            words = line.split()
            count = word_count(words)
            label,score = class_check(count, w,0)
            answer.write('{}\t{}'.format(label,line))
