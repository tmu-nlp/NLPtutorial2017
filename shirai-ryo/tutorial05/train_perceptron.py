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
