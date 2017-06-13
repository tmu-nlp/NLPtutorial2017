from collections import defaultdict
train_f = '../../data/titles-en-train.labeled'
model_f = 'model_file.txt'

def PREDICT_ONE(w, phi):
    score = 0
    for name, value in phi.items():
        if name in w:
            score += value * w[name]
    if score >= 0:
        return 1
    else:
        return -1

def CREATE_FEATURES(x):
    phi = defaultdict(lambda: 0)
    words = x.strip().split()
    for word in words:
        phi[word] += 1
    return phi

def UPDATE_WEIGHTS(w, phi, y):
    for name, value in phi.items():
        w[name] += int(value) * y

if __name__ == '__main__':
    w = defaultdict(lambda: 0)
    epoch = 20
    for i in range(epoch):
        with open('../../data/titles-en-train.labeled', 'r') as f:
            for line in f:
                y, x = line.strip().split('\t')
                phi = defaultdict(lambda: 0)
                for k, v in CREATE_FEATURES(x).items():
                    phi[k] = v
                n_y = PREDICT_ONE(w, phi)
                if n_y != int(y):
                    UPDATE_WEIGHTS(w, phi, int(y))

    with open(model_f, 'w') as f_m:
        for k, v in w.items():
            print(k)
            f_m.write('{} {}\n'.format(k, v))
