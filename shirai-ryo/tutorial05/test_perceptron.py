from train_perceptron import CREATE_FEATURES, PREDICT_ONE
from collections import defaultdict

with open("model_file.word") as text:
    w = defaultdict(int)
    for line in text:
        words = line.strip().split()
        w[words[0]] = int(words[1])


with open("../../data/titles-en-test.word") as text,open('my_answer.labeled', 'w') as answer:
    for line in text:
        phi = CREATE_FEATURES(line)
        yy = PREDICT_ONE(w, phi)
        answer.write(str(yy) + "\n")
