import numpy as np
import pickle
from collections import defaultdict

lam = 0.01

#確率が一番高いインデックスyを探す
#0番目からどんどん見ていく感じっぽい
def FIND_BEST(p):
    y = 0
    for i in range(1, len(p)):
        if p[i] > p[y]:
            y = i
    return y


def CREATE_ONE_HOT(id_, size):
    vec = np.zeros(size)
    vec[id_] = 1
    return vec

def softmax(x):
    x = x - max(x)
    y = np.exp(x) / sum(np.exp(x))
    return y

def forward_rnn(w_rx, w_rh, b_r, w_oh, b_o, x):
    h = [0] * len(x) # 隠れ層の値
    p = [0] * len(x) # 出力の確率分布の値
    y = [0] * len(x) # 出力の確率分布の値
    for t in range(len(x)):
        if t > 0:
            h[t] = np.tanh(np.dot(w_rx, x[t]) + np.dot(w_rh, h[t-1]) + b_r)
        else:
            h[t] = np.tanh(np.dot(w_rx, x[t]) + b_r)
        p[t] = softmax(np.dot(w_oh, h[t]) + b_o)
        y[t] = FIND_BEST(p[t])
    return h, p, y



def gradient_rnn(w_rx, w_rh, b_r, w_oh, b_o, x, h, p, y_):
    dw_rx = np.zeros((2, len(x_ids)))
    dw_rh = np.zeros((2, 2))
    db_r = np.zeros(2)
    dw_oh = np.zeros((len(y_ids), 2))
    db_o = np.zeros((len(y_ids)))
    delta_d_r = np.zeros(len(b_r))
    for t in reversed(range(len(x))):
        p_ = CREATE_ONE_HOT(y_[t], len(y_ids))
        delta_d_o = p_ - p[t]
        dw_oh += np.outer(delta_d_o, h[t])
        db_o += delta_d_o
        delta_r = np.dot(delta_d_r, w_rh) + np.dot(delta_d_o, w_oh)
        delta_d_r = delta_r * (1 - h[t]**2)
        dw_rx += np.outer(delta_d_r, x[t])
        db_r += delta_d_r
        if t != 0:
            dw_rh += np.outer(delta_d_r, h[t-1])
    return dw_rx, dw_rh, db_r, dw_oh, db_o


def UPDATE_WEIGHT(w_rx, w_rh, b_r, w_oh, b_o, dw_rx, dw_rh, db_r, dw_oh, db_o, lam):
    # d はデルタ
    w_rx += lam * dw_rx
    w_rh += lam * dw_rh
    b_r += lam * db_r
    w_oh += lam * dw_oh
    b_o += lam * db_o


if __name__ == "__main__":
    x_ids = defaultdict(lambda:len(x_ids))
    y_ids = defaultdict(lambda:len(y_ids))
    for line in open("../../data/wiki-en-train.norm_pos"):
        wordtags = line.strip().split()
        for word_tag in wordtags:
            wordtag = word_tag.split("_")
            x_ids[wordtag[0]]
            y_ids[wordtag[1]]
    feat_lab = []
    x_list = list()
    y_list = list()
    feat_lab.append((x_list, y_list))
    for line in open("../../data/wiki-en-train.norm_pos"):
        wordtags = line.strip().split()
        for word_tag in wordtags:
            wordtag = word_tag.split("_")
            x_list.append(CREATE_ONE_HOT(x_ids[wordtag[0]], len(x_ids)))
            y_list.append(y_ids[wordtag[1]])
    w_rx = (np.random.rand(2, len(x_ids)) - 0.5) * 0.02
    w_rh = (np.random.rand(2, 2) - 0.5) * 0.02
    b_r = np.zeros(2)
    w_oh = (np.random.rand(len(y_ids), 2) - 0.5) * 0.02
    b_o = np.zeros(len(y_ids))
    epoch = 10
    for i in range(epoch):
        for x, y_ in feat_lab:
            h, p, y = forward_rnn(w_rx, w_rh, b_r, w_oh, b_o, x)
            dw_rx, dw_rh, db_r, dw_oh, db_o = gradient_rnn(w_rx, w_rh, b_r, w_oh, b_o, x, h, p, y_)
            UPDATE_WEIGHT(w_rx, w_rh, b_r, w_oh, b_o, dw_rx, dw_rh, db_r, dw_oh, db_o, lam)

    with open('weight_file.dump', 'wb') as w_file:
        net = (w_rx, w_rh, b_r, w_oh, b_o)
        pickle.dump(net, w_file)
    with open('id_file.dump', 'wb') as id_file:
        ids = (dict(x_ids), dict(y_ids))
        pickle.dump(ids, id_file)
