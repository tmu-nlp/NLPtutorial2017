import numpy as np
import dill
from collections import defaultdict
#import random
from pprint import pprint


model_file = 'rnn.model'


def create_one_hot(size, k):
    v = np.zeros(size, dtype=int)
    v.put(k, 1)
    return v


def init_nn():
    w_rx = np.array(np.random.rand(2, len(x_ids) ) - 0.5)/100
    w_rh = np.array(np.random.rand(2, 2) - 0.5)/100
    b_r = np.zeros(2)
    w_oh = np.array(np.random.rand( len(y_ids), 2) - 0.5)/100
    b_o = np.zeros(len(y_ids))
    nn = [w_rx, w_rh, b_r, w_oh, b_o]
    return nn


def forward_rnn(nw, x):
    ''' 
    nw[0] = wrx
    nw[1] = wrh
    nw[2] = br
    nw[3] = woh
    nw[4] = bo
    nw[5] = wrx
    '''

    h = [0] * len(x)
    p = [0] * len(x)
    y = [0] * len(x)

    # y = [0 for i in range(len(x))]
    for t in range(len(x)):
        if t > 0:
            h[t] = np.tanh(np.dot(nw[0], x[t]) + np.dot(nw[1], h[t-1]) + nw[2])
        else:
            h[t] = np.tanh(np.dot(nw[0], x[t]) + nw[2])
        p[t] = softmax(np.dot(nw[3], h[t]) + nw[4])
        y[t] = np.argmax(p[t])
    return h, p, y


def softmax(w):
    _exp = np.exp(np.array(w))
    return _exp / np.sum(_exp)


def gradient_rnn(nw, x, h, p, y_correct):
    delta_nn = init_nn()
    d_rp = np.zeros(len(nw[2]))
    for t in range(len(x)-1, -1, -1):
        d_op =  y_correct[t] - p[t]
        delta_nn[3] += np.outer(d_op, h[t])
        delta_nn[4] += d_op
        d_r = np.dot(d_rp, nw[1]) + np.dot(d_op, nw[3])
        d_rp = d_r * (1 - h[t]**2)
        delta_nn[0] += np.outer(d_rp, x[t])
        delta_nn[2] += d_rp
        if t != 0:
            delta_nn[1] += np.outer(d_rp, h[t-1])

    return delta_nn

from operator import add
def update_weights(nw, delta_nw, _eta):
    # map(add, nw, map(add, nw, _eta * delta_nw))
    # _nw = np.array(nw)
    # _delta_nw = np.array(delta_nw)
    # _nw += _eta * _delta_nw
    # nw = _nw.tolist()
    for idx, dnw in enumerate(delta_nw):
        nw[idx] += _eta * dnw

    


import sys
if __name__ == '__main__':

    max_epoch = 1000
    eta = 0.01

    x_ids = defaultdict(lambda: len(x_ids))
    y_ids = defaultdict(lambda: len(y_ids))
    
    train_data = "../../test/05-train-input.txt"
    try:
        if sys.argv[1] == r:
            train_data = "../../data/wiki-en-train.norm_pos"
    except:
        pass
    
    ''' 訓練データをロードし、単語とタグの辞書をつくる '''
    with open(train_data) as f:
        for line in f:
            wordtags = line.strip().split(" ")
            for wordtag in wordtags:
                word, tag = wordtag.split("_")
                x_ids[word.lower()]
                y_ids[tag]
    

    ''' 訓練データをロードし、単語とタグの1hotベクトルを生成しfeat_labをつくる '''
    with open(train_data) as f:
        feat_lab = list()
        for line in f:
            word_list = list()
            tag_list = list()
            wordtags = line.strip().split(" ")
            for wordtag in wordtags:
                word, tag = wordtag.split("_")
                '''一行の単語とタグを1-hot vectorで表現'''
                word_list.append(create_one_hot(len(x_ids), x_ids[word.lower()] ))
                tag_list.append(create_one_hot(len(y_ids), y_ids[tag]))
            '''一行分をfeatlabへ'''
            feat_lab.append([word_list, tag_list])
    
    net = init_nn()
    
    
    # main loop
    for epoch in range(max_epoch):
        print('epoch:{}/{}'.format(epoch+1, max_epoch))
        np.random.shuffle(feat_lab)
        ''' 単語と正解ラベルをひとつずつNNにいれて誤差逆伝搬させ重みを更新する'''
        for word, corr_tag in feat_lab:
            h, p, y = forward_rnn(net, word)
            d_nn = gradient_rnn(net, word, h, p, corr_tag)
            update_weights(net, d_nn, eta)
    
    
    # dump to file
    with open(model_file,'wb') as model_f:
        dill.dump((x_ids, y_ids, net), model_f)
    
