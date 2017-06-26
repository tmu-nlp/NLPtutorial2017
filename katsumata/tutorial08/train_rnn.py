import numpy as np
import pickle
from collections import defaultdict
import random

def softmax(wonda):
    #wondaはx
    return np.exp(wonda) / np.sum(np.exp(wonda))

def create_onehot(arg, ids):
    #ここでvecのlengthは決定しておく
    vec = np.zeros(len(ids))
    vec[ids[arg]] = 1
    return vec    

def update_weights(network, delta, lam):
    for w, d in zip(network, delta):
        w += lam*d

def find_max(p):
    y = 0
    for i in range(len(p)):
        if p[i] > p[y]:
            y = i
    return y #確率が最も高いindexを返す        

def forward_rnn(network, x):
    h = list() #hidden layer
    p = list() #probability
    y = list() #prediction
    w_rx, w_rh, w_oh, b_r, b_o = network

    for t in range(len(x)):
        if t > 0:
            h.append(np.tanh(np.dot(w_rx, x[t]) + np.dot(w_rh, h[t-1]) + b_r))
        else:
            h.append(np.tanh(np.dot(w_rx, x[t]) + b_r))
        p.append(softmax(np.dot(w_oh, h[t]) + b_o))
        y.append(find_max(p[t]))
    return h, p, y    
def backward_rnn(network, x, h, p, p_co, err_c):
    d_weight_rx = np.zeros((2, len(x_ids)))
    d_weight_rh = np.zeros((2,2))
    d_weight_oh = np.zeros((len(y_ids), 2)) 
    d_b_r = np.zeros(2)
    d_b_o = np.zeros(len(y_ids))
    w_rx, w_rh, w_oh, b_r, b_o = network

    delta_r_ = np.zeros(len(b_r)) #t+1から伝搬されるエラー
    for t in reversed(range(len(x))):
        #p_ = create_onehot(y_co[t]) #なんか変
        delta_o_ = (p_co[t] - p[t])  #出力層のエラー(多分tつけんと麻酔)
        """
        print ('正解ラベル')
        print (p_co[t])
        print ('softmaxされた確率')
        print (p[t])
        print ('差分')
        print (delta_o_)
        """
        err_c += delta_o_
        d_weight_oh += np.outer(delta_o_, h[t])
        d_b_o += delta_o_                        #出力重み勾配
        delta_r = np.dot(delta_r_, w_rh) + np.dot(delta_o_, w_oh) #逆伝搬
        #delta_r = np.dot(delta_o_, w_oh) #逆伝搬
        delta_r_ = delta_r * (1 - h[t] ** 2) #tanhの重み勾配
        d_weight_rx += np.outer(delta_r_, x[t]) 
        d_b_r += delta_r_ #hidden layer重み勾配
        if t != 0:
            d_weight_rh += np.outer(delta_r_, h[t-1])
        
    ddddd = [d_weight_rx, d_weight_rh, d_weight_oh, d_b_r, d_b_o]        
    return ddddd, err_c

if __name__ == '__main__':
    """
    やることまとめ
    単語をid化,train_dataからラベルと、テキストから素性を持ってくる
    ネットワーク、つまり重みを初期化する
    持ってきたラベルと素性を一つずつ(for)持ってきてforward,backwardそして重み更新して終わり
    ここまで前回、ここからRNN
    瀕死推定のため少しids等が異なる、forwardが一つ前のhidden unitを見る、勾配がhidden unitを通ってback
    この3点が大きく異なるはず
    """
    l = 10
    train_f = '../../data/wiki-en-train.norm_pos'
    #train_f='../../test/05-train-input.txt'
    feat_lab = list()
    layer_num = 2
    lam = .01
    x_ids = defaultdict(lambda:len(x_ids))
    y_ids = defaultdict(lambda:len(y_ids))

    #素性の数を数えて重みの初期化を行う
    #出現単語、瀕死にidsを振る,その後初期化とか
    with open(train_f, 'r') as train_init:
        for line in train_init:
            words = line.split()
            for word in words:
                x, y = word.split('_')
                x_ids[x]
                y_ids[y]
    with open(train_f, 'r') as train_init:
        for line in train_init:
            x_list = list()
            y_list = list()
            words = line.split()
            for word in words:
                x, y = word.split('_')
                x_list.append(create_onehot(x, x_ids)) #単語列
                y_list.append(create_onehot(y, y_ids)) #瀕死列
            #feat_labに一文単位で各単語の素性と正解ラベルを入れていく
            feat_lab.append((x_list, y_list))
        #ここからネットワークの初期化
        weight_rx = (np.random.rand(2, len(x_ids)) - .5) * .2 * .1
        weight_rh = (np.random.rand(2,2) - .5) * .2*.1
        weight_oh = (np.random.rand(len(y_ids), 2) - .5) * .2 * .1
        b_r = np.zeros(2)
        b_o = np.zeros(len(y_ids))
        network = [weight_rx, weight_rh, weight_oh, b_r ,b_o]
    #ここから学習    
    for i in range(l):
        random.shuffle(feat_lab)
        err_count = 0
        print ('epoch {}'.format(i))
        for x, y_correct in feat_lab:
            h, p, y_predict = forward_rnn(network, x) 
            delta, err_count = backward_rnn(network, x, h, p, y_correct, err_count)
            update_weights(network, delta, lam)
        #print (err_count)
    with open('network.dump', 'wb') as net_f:
        pickle.dump(network, net_f)
    
    with open('x_ids.dump', 'wb') as ids_f1:
        pickle.dump(dict(x_ids), ids_f1)
    
    with open('y_ids.dump', 'wb') as ids_f2:
        pickle.dump(dict(y_ids), ids_f2)
    
