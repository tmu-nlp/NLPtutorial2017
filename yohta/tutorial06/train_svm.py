#予測
from collections import defaultdict
import math
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
    phi = defaultdict(lambda :0)
    words = x.strip().split()
    for word in words:
        phi['UNI:' + word] += 1
    return phi
"""
def update_weights(w,phi,y):
    c = 0.0001
    for name,value in w.items():
        if abs(value) < c:
            w[name] = 0
        else:
            w[name] -= sign(value) * c
    for name,value in phi.items():
        w[name] += value * y
    return w
"""
def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

sigmoid = defaultdict(lambda :0)

def sigm(x,word):
    if x >= 0:
        sigmoid[word] += (math.exp(x)/(1+math.exp(x))**2)
        return sigmoid[word]
    else:
        sigmoid[word] -= (math.exp(x)/(1+math.exp(x))**2)
        return sigmoid[word]


if __name__ == '__main__':
    w = defaultdict(lambda :0)
    l = 20 #iteration? : 試行数？
    margin = 20
    c = 0.0001
    for i in range(l):
        with open('../../data/titles-en-train.labeled','r') as t_f:
            for line in t_f:
                phi = defaultdict(lambda :0)
                y,x = line.strip().split('\t') #y is int , x is words
                y = float(y)
                for word,value in create_features(x).items():
                    phi[word] = value
                    val = w[word] * phi[word] * y
                    if val <= margin:
                        if abs(w[word]) < c:
                            w[word] = 0
                        else:
                            w[word] += sigm(w[word],word) * c
#                            w[word] -= sign(w[word]) * c
                        w[word] += phi[word] * y
    with open('model_file.txt','w') as m_f:
#        for line in m_f:
        for word,value in w.items():
            m_f.write('{}\t{}\n'.format(word,value))
