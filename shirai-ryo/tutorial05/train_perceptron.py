# メモ
# 何をやりたいか（なんとなくわかる範囲で）
# Xという事例において、ある単語が何回出てくるか
# そして、その単語には人物かどうかの評価の値がついている
# （プラスなら人物、マイナスなら人物じゃない）
# 単語　×　評価の値を全て足し合わせたもの＝重み付き和
# 0以上ならyes(+1) 、0未満なら(-1)。

from collections import defaultdict


def PREDICT_ONE(w, phi): # 一文が人か人じゃないか
    score = 0
    for key, value in phi.items():
        if key in w:
            score += value * w[key]
    if score >= 0:
        return 1
    else:
        return -1

def CREATE_FEATURES(x): #　単語ごとの出現回数
    phi = defaultdict(int)
    words = x.split()
    for word in words:
        phi[word] += 1 #「 UNI: 」を追加して1-gramを表す
    return phi

def UPDATE_WEIGHT(w, phy, y): #重みの更新
    for name, value in phi.items():
        w[name] += int(value) * y


if __name__ == "__main__":
    w = defaultdict(int)
    epoch = 10
    with open("../../data/titles-en-train.labeled") as text:
        for line in text:
            line = line.split('\t')
            y = int(line[0])
            x = line[1]
            phi = CREATE_FEATURES(x)
            yy = PREDICT_ONE(w, phi)
            if yy != y:
                UPDATE_WEIGHT(w, phi, y)
    with open('model_file.word','w') as model:
        for word,predict in w.items():
            model.write('{} {}\n'.format(word,predict))


# word_dict = defaultdict(int)
#
# with open("../../data/titles-en-train.labeled") as text:
#     #疑似コードでいう、model_file？
#     for line in text:
#         words = line.split()
#         for word in words:
#             word_dict[word] += 1

# with open("model_file.word", "w") as text:
#     for word, count in word_dict.items():
#         text.write(word + " " + str(count) + "\n")

#これユニグラムのやつだった（死）
