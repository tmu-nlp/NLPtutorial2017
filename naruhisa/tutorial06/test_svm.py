from collections import defaultdict
from train_svm import PREDICT_ONE, CREATE_FEATURES
test_f = '../../data/titles-en-test.word'
model_f = 'model_file.txt'

def PREDICT_ALL(model_f, input_f):
    w = defaultdict(int)
    with open(model_f, 'r') as f1, open(input_f, 'r') as f2:
        for line in f1:
            words = line.strip().split()
            w[words[0]] = float(words[1])
        for x in f2:
            phi = dict()
            for k, v in CREATE_FEATURES(x).items():
                phi[k] = v
            n_y = PREDICT_ONE(w, phi)
            yield('{}\t{}'. format(n_y, x))

if __name__ == '__main__':
    with open('my_answer.labeled', 'w') as f:
        for line in PREDICT_ALL(model_f, test_f):
            f.write(line)
