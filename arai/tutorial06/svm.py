from collections import defaultdict
train_f = '../../data/titles-en-train.labeled'
model_f = 'model_file.txt'
test_f = '../../data/titles-en-test.word'

def predict_one(w, phi):
    score = 0
    for name, value in phi.items():
        if name in w:
            score += value * w[name]
    return score

def create_features(x):
    phi = defaultdict(int)
    words = x.split()
    for word in words:
        phi[word] += 1
    
    return phi

def update_weights(w, phi, y, c,l, last):
    for name,value in phi.items():
        if l != last[name]:
            c_size = c * (l - last[name])
            if abs(w[name]) <= c_size:
                w[name] = 0
            else:
                w[name] -= sign(w[name]) * c_size
            last[name] = l
        w[name] += value * int(y)

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

if __name__ == '__main__':
    w = defaultdict(lambda: 0.0)
    c = 0.0001
    l = 25
    margin = 50
    last = defaultdict(lambda: 0)
    #"""
    for i in range(l):
        #print(i)
        with open(train_f) as t_f:
            for line in t_f:
                #print(line)
                y, x = line.strip().split('\t')
                phi = create_features(x)
                score = predict_one(w, phi) * int(y)
                if score <= margin:
                    update_weights(w, phi, y, c, i, last)
    with open(model_f, 'w') as m_f:
        for key, value in sorted(w.items()):
            m_f.write('{} {}\n'.format(key, value))
    #"""

    with open(test_f) as test , open(model_f) as model:
        w = defaultdict(lambda: 0.0)
        for line in model:
            words = line.strip().split()
            w[words[0]] = float(words[1])
        for x in test:
            phi = create_features(x)
            if predict_one(w, phi) >=0:
                y_ = 1
            else:
                y_ = -1
            printline = ( str(y_) + '\t' + str(x)).strip('\n')
            print(printline)
 
            

    

