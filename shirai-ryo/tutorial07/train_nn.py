import numpy as np
import pickle
# pickleという単語を使ってはいるもののやっていることはシリアライズ(直列化)、デシリアライズ(非直列化)です。
# pickle.dump([オブジェクト], [ファイル]):
#       指定したファイルにオブジェクトを書き出す
# pickle.load([ファイル]):
#       指定したファイルからオブジェクトを取り出す

# "rb"はバイナリの形で読み込み、"wb"はバイナリの形で書き込み

# numpy.dot() で内積を求めることができる
# numpy.outer() で外積を求めることができる


# from collections import defaultdict


# パーセプトロンの予測
# def PREDICT_ONE(w, phi):
#     score = np.dot(w,phi)
#     if score[0] >= 0:
#         return 1
#     else:
#         return -1


# 素性のID化
# ids = defaultdict(lambda: len(ids))
#
# def CREATE_FEATURES(x):
#     phi = [0] * len(ids)
#     words = x.split()
#     for word in words:
#         phi[ids["UNI" + word]] += 1
#     return phi

#素性の初期化
#ゼロで初期化 w = np.zeros(len(ids))
#[-0.5, 0.5]でランダムに初期化
# w = np.random.rand(2, len(ids)) - 0.5



def forward_nn(network, phi_zero):
    phi = [phi_zero]
    for i in range(len(network)):
        weight = network[i][1:]
        bias = network[i][0]
        phi.append(np.tanh(np.dot(weight.T, phi[i].T) + bias))
    return phi


def backward_nn(network, phi, yy):
    J = len(network)
    delta = [0] * (J+1)
    delta[J] = yy - phi[J]
    delta_d = [0] * (J+1)
    for i in reversed(range(J)):
    # range(始まり, 終わり, ステップ)
    # ステップをマイナスにすれば、大きい数字→小さい数字と見ていける
        delta_d[i+1] = delta[i+1] * (1 - phi[i+1]**2)
        weight = network[i][1:]
        delta[i] = np.dot(delta_d[i+1], weight.T)
    return delta_d


def update_weight(network, phi, delta_d, Lambda):
    for i in range(len(network)):
        network[i][1:] += Lambda * np.outer(delta_d[i + 1], phi[i]).T
        network[i][0] += Lambda * delta_d[i + 1]



if __name__ == "__main__":
    ids = dict()
    feat_lab = list()
    Lambda = 0.1
    for line in open("../../data/titles-en-train.labeled"):
        y, x = line.strip('\n').split('\t') # y, xがよくわらかん
        for word in x.split():
            if word not in ids:
                ids[word] = len(ids)


    for line in open('../../data/titles-en-train.labeled', 'r'):
        polarity, sentence = line.strip('\n').split('\t')
        features = np.zeros(len(ids))
        for word in sentence.split():
            features[ids[word]] += 1
        feat_lab.append((features, int(polarity)))
        # 0 < rand < 1  ->  0 < 0.2rand < 0.2  ->  -0.1 < 0.2rand - 0.1 < 0.1
    network = [np.random.rand(len(ids) + 1, 2) * 0.2 - 0.1, np.random.rand(3, 1) * 0.2 - 0.1]
    # numpy.random.rand() で 0〜1 の一様乱数を生成する。引数を指定すれば複数の乱数を生成できる。
    # 乱数の範囲を変えたい場合は後からベクトル演算をすれば良い。
    # """
    # from numpy.random import *
    #
    # rand()      # 0〜1の乱数を1個生成
    # rand(100)   # 0〜1の乱数を100個生成
    # rand(10,10) # 0〜1の乱数で 10x10 の行列を生成
    #
    # rand(100) * 40 + 30 # 30〜70の乱数を100個生成
    # """
    epoch = 3
    for i in range(epoch):
        for phi_zero, polarity in feat_lab:
            phi = forward_nn(network, phi_zero)
            delta_prime = backward_nn(network, phi, polarity)
            update_weight(network, phi, delta_prime, Lambda)

    with open('weight_file.dump', 'wb') as w_file:
        pickle.dump(network, w_file)
    with open('id_file.dump', 'wb') as id_file:
        pickle.dump(ids, id_file)





# #学習を行う
#     ids_dict = defaultdict(int)
#     with open("../../data/titles-en-train.labeled") as text:
#         for line in text:
#             line = line.split('\t')
#             y = int(line[0])
#             x = line[1]
#             phi = CREATE_FEATURES(x)
#             yy = PREDICT_ONE(w, phi)
#             if yy != y:
#                 UPDATE_WEIGHT(w, phi ,y)
#                 print()


#pickleとpickle.dumpについてググる
