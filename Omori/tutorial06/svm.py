from collections import defaultdict
import math
import sys
import random

def train_svm(train_file):
    with open(train_file) as f:
        weight = defaultdict(int)
        train_list = list(f)
        margin = 20
        for I in range(10):
            random.shuffle(train_list)
            for line in train_list:
                label, sent = line.strip().split('\t')
                phi = create_features(sent)
                _, score, weight = predict_one(weight, phi)
                val = score * int(label) 
                if val <= margin:
                    weight = update_weight(weight, phi, int(label))

    return weight

def create_features(sent):  # unigram model
    phi = defaultdict(int)
    words = sent.split()
    for word in words:
        phi["UNI:"+word] += 1
    return phi

def predict_one(weight, phi):
    score = 0
    for word, value in phi.items():
        if word in weight:
            score += value * weight[word]
    if score >= 1:
        return 1, score, weight
    else:
        return -1, score, weight

def update_weight(weight, phi, label):
    c = 0.0001
    for word, value in weight.items():
        if math.fabs(value) < c:
            weight[word] = 0
        else:
            weight[word] -= sign(value) * c  # 0に近づける方向
    for word, value in phi.items():
        weight[word] += value*label
    return weight

def sign(value):
    if value > 0:
        return 1
    elif value == 0:
        return 0
    else:
        return -1

def test_svm(test_file, weight):
    with open(test_file) as f:
        for line in f:
            sent = line.strip()
            phi = create_features(sent)
            y, _, _ = predict_one(weight, phi)
            print(y)

if __name__ == "__main__":
    w = train_svm(sys.argv[1])
    #print(len(w))
    #for k, v in w.items():
    #    print(k,v)
    test_svm(sys.argv[2], w)

    # --margin 20 --c 0.0001 Accuracy = 93.269571% 


