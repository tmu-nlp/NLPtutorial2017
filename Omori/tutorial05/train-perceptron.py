from collections import defaultdict
import sys
import random

def train(train_file):
    with open(train_file) as f:
        weight = defaultdict(int)
        train_list = list(f)
        for I in range(10):
            random.shuffle(train_list)
            for line in train_list:
                label, sent = line.strip().split('\t')
                phi = create_features(sent)
                y, weight = predict_one(weight, phi)
                if y != int(label):
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
    if score >= 0:
        return 1, weight
    else:
        return -1, weight

def update_weight(weight, phi, label):
    for word, value in phi.items():
        weight[word] += value*label
    return weight

if __name__ == "__main__":
    w = train(sys.argv[1])
    print(len(w))
    for k, v in w.items():
        print(k,v)
