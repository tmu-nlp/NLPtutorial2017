from collections import defaultdict
train_f = '../../data/titles-en-train.labeled'
model_f = 'model_file.txt'

    
def predict_one(w, phi):
    score = 0
    for name, value in phi.items():
        if name in w:
            score += value * w[name]
    if score >= 0:
        return 1
    else:
        return -1

def create_features(x):
    phi = defaultdict(int)
    words = x.split()
    for word in words:
        phi[word] += 1

    return phi

def update_weights(w, phi, y):
    for name, value in phi.items():
        w[name] += value * y

if __name__ == '__main__':
    w = defaultdict(int)
    l = 10
    for i in range(l):
        with open(train_f) as t_f:
            for line in t_f:
                y, x = line.strip().split('\t')
                phi = create_features(x)
                y_ = predict_one(w, phi)
                if y_ != int(y):
                    update_weights(w, phi, int(y))
    with open(model_f, 'w') as m_f:
        for key, value in w.items():
            m_f.write('{} {}\n'.format(key, value))

