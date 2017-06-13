import sys
from collections import defaultdict

class WEIGHT_UNI():
    def __init__(self):
        self.w = defaultdict(int)

    def updata(self,count,c):
        for word in count.keys():
            self.w[word] += count[word]*c

    def weight_get(self):
        return self.w

class WEIGHT_BI():
    def __init__(self):
        self.w = defaultdict(int)

    def updata(self,count,c):
        for word in count.keys():
            self.w[word] += count[word]*c

    def weight_get(self):
        return self.w 

def class_check(count_uni_class,count_bi_class,weight_uni_class,weight_bi_class):
    score = 0
    for word_class in count_uni_class.keys():
        score += count_uni_class[word_class]*weight_uni_class[word_class]
    for word_class_bi in count_bi_class.keys():
        score += count_bi_class[word_class_bi]*weight_bi_class[word_class_bi]
    if score >= 0:
        return 1
    else:
        return -1

def word_count(words_c):
    count_c = defaultdict(int)
    for word in words_c:
        count_c[word] += 1
    return count_c

def word_count_bi(words_c_bi):
    count_c_bi = defaultdict(int)
    for i in range(len(words_c_bi)-1):
        count_c_bi[words_c_bi[i]+' '+words_c_bi[i+1]] += 1
    return count_c_bi

def fit(weight_uni,weight_bi):
    with open(sys.argv[1]) as train:
        for line in train:
            label, words = line.split('\t')
            label = int(label)
            words = words.lower().split()
            count = word_count(words)
            count_bi = word_count_bi(words)
            class_now = class_check(count,count_bi,weight_uni.weight_get(),weight_bi.weight_get())
            if class_now != label:
                weight_uni.updata(count,label)
                weight_bi.updata(count_bi,label)

if __name__ == '__main__':
    epoch = 40
    w_uni = WEIGHT_UNI()
    w_bi = WEIGHT_BI()
    for i in range(epoch):
        fit(w_uni,w_bi)
    with open('model_bi_2.txt','w') as model:
        for word, prob in w_uni.weight_get().items():
            if prob != 0:
                model.write('uni:=:{}:=:{}\n'.format(word,prob))
        for word, prob in w_bi.weight_get().items():
            if prob != 0:
                model.write('bi:=:{}:=:{}\n'.format(word,prob))

