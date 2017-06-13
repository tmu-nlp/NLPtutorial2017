from collections import defaultdict
from train import create_feature

def predict_one(w, phi):
    score = 0
    for name, value in phi.items():# score = w*φ(x)
        if name in w:
            score += float(value) * w[name]
    if score >= 0:
        return 1
    else:
        return -1
def predict_all(model_file, input_file):
    w = defaultdict(int)
    for line in open(model_file):
        name, value = line.split("\t")
        w[name] = float(value)
    with open("my_answer.word", "w") as output:
        for x in open(input_file):
            phi = create_feature(x)
            y_ = predict_one(w,phi)
            print(y_,file=output)

input_file = "../../data/titles-en-test.word"
model_file = "model.txt"
predict_all(model_file,input_file)
