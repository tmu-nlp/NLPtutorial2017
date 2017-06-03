import collections

def CREATE_FEATURES(txt):
    phi = collections.defaultdict(lambda: 0)
    words = txt.split()
    for word in words:
        phi['UNI:' + word] += 1
    return phi

def PREDICT_ONE(w, phi):
    score = 0
    for name, value in phi.items():
        if name in w.keys():
            score += value*w[name]
    if score >= 0:
        return 1
    else:
        return -1

def UPDATE_WEIGHTS(w, phi, y):
    for name, value in phi.items():
        w[name] += value*y

def train(data_train, w):
    for i,line in enumerate(data_train):
        y, txt = line.split('\t')
        y = int(y)
        txt = txt.lower()
        phi = CREATE_FEATURES(txt)
        y_predict = PREDICT_ONE(w, phi)
        if y_predict != y:
            UPDATE_WEIGHTS(w, phi, y)

def train_epoch(epoch, path_data_train):
    w = collections.defaultdict(lambda: 0)
    for i in range(epoch):
        with open(path_data_train, 'r') as data_train:
            train(data_train, w)
    return w

if __name__ == '__main__':
    epoch = 1
    path_data_train = '../../data/titles-en-train.labeled'
    with open('train_perceptron.txt', 'w') as data_out:
        print(train_epoch(epoch, path_data_train), file=data_out)
