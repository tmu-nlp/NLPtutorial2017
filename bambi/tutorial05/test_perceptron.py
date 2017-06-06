
from collections import defaultdict
from train_perceptron import predict_one, create_feature
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
