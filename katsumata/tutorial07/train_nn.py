import numpy as np
import pickle
from collections import defaultdict

ids = defaultdict(lambda:len(ids))

def create_features(x):
    #ここでphiのlengthは決定しておく
    phi = [0 for i in range(len(ids))]
    words = x.split()
    for word in words:
        phi[ids['UNI:'+word]] += 1
    return phi    

def update_weights(network, phiS, delta_, lam):
    for i in range(len(network)):
        w, b = network[i]
        network[i][0] += lam*np.outer(delta_[i+1], phiS[i]) #こっちが重み
        network[i][1] += lam*delta_[i+1] #こっちがバイアス

def forward_nn(network, phi_0):
    phiS = [0 for i in range(len(network)+1)]
    phiS[0] = phi_0 #角層の値
    for i in range(1, len(network)+1):
        w, b = network[i-1]
        #前の値に基づいて値を計算
        phiS[i] = np.tanh(np.dot(w, phiS[i-1]) +b)
    return phiS    

def backward_nn(net, phiS, y):
    J = len(net)
    delta = [0 for i in range(J)]
    delta.append(np.array([y-phiS[J][0]]))
    print (y - phiS[J][0])
    delta_ = [0 for i in range(J+1)]
    for i in reversed(range(J)):
        delta_[i+1] = delta[i+1]*(1-phiS[i+1]**2)
        w, b = net[i]
        delta[i] = np.dot(delta_[i+1], w)
    return delta_    

if __name__ == '__main__':
    """
    やることまとめ
    単語をid化,train_dataからラベルと、テキストから素性を持ってくる(多分バッチ)
    ネットワーク、つまり重みを初期化する
    持ってきたラベルと素性を一つずつ(for)持ってきてforward,backwardそして重み更新して終わり

    """
    l = 2 
    train_f = '../../data/titles-en-train.labeled'
    #test_f = '../../data/titles-en-test.word'
    test_f = '../../test/03-train-input.txt'
    #weight_f = 'weight_file.nn'
    #id_f = 'id_file.nn'
    feat_lab = list()
    layer_num = 2
    lam = .1 
    #素性の数を数えて重みの初期化を行う
    #出現単語にidsを振る,その後初期化とか
    with open(train_f, 'r') as train_init:
        for line in train_init:
            y, x = line.strip().split('\t')
            words = x.split()
            for word in words:
                 ids['UNI:'+word]
    with open(train_f, 'r') as train_init:
        for line in train_init:
            y, x = line.strip().split('\t')
            y = int(y)
            #feat_labに一文の素性と正解ラベルを入れていく
            feat_lab.append((create_features(x), y))
        #ここからネットワークの初期化
        weight_0 = (np.random.rand(2,len(ids)) - .5) * .2 #(2*27190)
        b_0 = np.zeros(2)
        weight_1 = (np.random.rand(1,2) - .5)*.2
        b_1 = np.zeros(1)
        network = [[weight_0, b_0], [weight_1, b_1]]
    #ここから学習    
    for i in range(l):
        for phi_0, y in feat_lab:
            phiS = forward_nn(network, phi_0) #ここのphiSは各層のphiを要素とするもの
            delta_ = backward_nn(network, phiS, y)
            update_weights(network, phiS, delta_, lam)
    with open('network.dump', 'wb') as net_f:
        pickle.dump(network, net_f)
    
    with open('ids.dump', 'wb') as ids_f:
        pickle.dump(dict(ids), ids_f)
    
