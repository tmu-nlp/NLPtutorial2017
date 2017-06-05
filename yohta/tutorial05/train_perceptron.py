#予測
from collections import defaultdict
#def predict_all(model_file,input_file):
    #load w from model_file
    #model_fileは未作成：testプログラムにおいて読み込む


def predict_one(w,phi):
    score = 0
    for name,value in phi.items():
        if name in w:
            score += value * w[name]
    if score >= 0:
        return 1
    else:
        return -1

def create_features(x):
    phi = defaultdict(int)
    words = x.strip().split()
    for word in words:
        phi['UNI:' + word] += 1
    return phi

def update_weights(w,phi,y):
    for name,value in phi.items():
        w[name] += int(value) * y

if __name__ == '__main__':
    w = defaultdict(int)
#    l = ?? #iteration? : 試行数？
#    for i in range(l):
    with open('../../data/titles-en-train.labeled','r') as t_f:
        for line in t_f:
            phi = defaultdict(int)
            y,x = line.strip().split('\t') #y is int , x is words
            for word,value in create_features(x).items():
                phi[word] += value
                y_ = predict_one(w,phi)
                if y_ != y:
                    update_weights(w,phi,int(y))
    with open('model_file.txt','w') as m_f:
#        for line in m_f:
        for word,value in w.items():
            m_f.write('{}\t{}\n'.format(word,value))
