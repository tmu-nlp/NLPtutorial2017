import sys
from collections import defaultdict

class WEIGHT():
    def __init__(self):
        self.w = defaultdict(int)

    def updata(self,count,c):
        for word in count.keys():
            self.w[word] += count[word]*c

    def weight_get(self):
        return self.w

def class_check(count_class, weight_class):
    score = 0
    for word_class in count_class.keys():
        score += count_class[word_class]*weight_class[word_class]
    if score >= 0:
        return 1
    else:
        return -1

def word_count(words_c):
    count_c = defaultdict(int)
    for word in words_c:
        count_c[word] += 1
    return count_c

def fit(weight_):
    with open(sys.argv[1]) as train:
        for line in train:
            label, words = line.split('\t')
            label = int(label)
            words = words.split()
            count = word_count(words)
            class_now = class_check(count, weight_.weight_get())
            if class_now != label:
                weight_.updata(count,label)

if __name__ == '__main__':
    epoch = 10
    w = WEIGHT()
    for i in range(epoch):
        fit(w)
    with open('model.txt','w') as model:
        for word, prob in w.weight_get().items():
            if prob != 0:
                model.write('{} {}\n'.format(word,prob))

