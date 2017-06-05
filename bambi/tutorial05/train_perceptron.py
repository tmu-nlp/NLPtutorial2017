
from collections import defaultdict
def predict_one(w, phi):
    score = 0
    for name, value in phi.items():# score = w*φ(x)
        if name in w:
            score += float(value) * w[name]
    if score >= 0:
        return 1
    else:
        return -1

def create_feature(x):
    phi = defaultdict(int)
    words =x.split()
    for word in words:
        phi["UNI:{}".format(word)] += 1 #We add “UNI:” to indicate unigrams
    return phi

def update_weight(w,phi,y):
    for name, value in phi.items():
        w[name] += float(value) * float(y)
if __name__ == "__main__":
    #input_file = "../test/03-train-input.txt"
    input_file = "../../data/titles-en-train.labeled"

    w = defaultdict(int)
    for line in open(input_file):
        y, x = line.split("\t")
        phi = create_feature(x)
        y_ = predict_one(w,phi)
        if y_ != y:
            update_weight(w,phi,y)
    with open("model.txt", "w") as output:
        for name, value in w.items():
            print("{}\t{}".format(name,value),file=output)
