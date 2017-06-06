from collections import defaultdict
from train_perceptron import predict_one, create_features

def predict_all():
    w = defaultdict(int)
    with open('../../data/titles-en-test.word') as i_f, open('model_file.txt') as m_f:
        for line in m_f:
            words = line.strip().split()
            w[words[0]] = int(words[1])
        for x in i_f:
            phi = create_features(x)
            y_ = predict_one(w, phi)
            printline = ( str(y_) + '\t' + str(x) ).strip('\n')
            print(printline)

if __name__ == '__main__':
    predict_all()



