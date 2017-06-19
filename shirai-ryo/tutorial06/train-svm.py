from collections import defaultdict
import math


def PREDICT_ONE(w, phi): # 一文が人か人じゃないか
    score = 0
    for key, value in phi.items():
        if key in w:
            score += value * w[key]
    # if score >= 0:
    return score
    #     return 1
    # else:
    #     return -1

def CREATE_FEATURES(x): #　単語ごとの出現回数
    phi = defaultdict(float)
    words = x.split()
    for word in words:
        phi[word] += 1
    return phi

def UPDATE_WEIGHT(w, phy, y, c): #重みの更新
    for name, value in w.items():
        if abs(value) < c:
            #abs()は絶対値をだす
            w[name] = 0
        else:
            w[name] -= sign(value) * c
        w[name] += int(value) * y
    for name, value in phi.items():
        w[name] += value * int(y)

def GETW(w, name, c, epoch, last):
    for name, value in phi.items():
        if epoch != last[name]: #重みが古くなっている
            c_size = c * (epoch - last[name])
            if abs(w[name]) <= c_size:
                w[name] = 0
            else:
                w[name] -= sign(w[name]) * c_size
            last[name] = epoch
        w[name] += value * int(y)

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

# def valval(w, phi):
#     score = 0
#     for key, value in phi.items():
#         if key in w:
#             score += value * w[key]
#     return score

if __name__ == "__main__":
    w = defaultdict(float)
    c = 0.0001
    epoch = 10
    margin = 50
    last = defaultdict(float)
    for i in range(epoch):
        with open("../../data/titles-en-train.labeled") as text:
            for line in text:
                line = line.split('\t')
                y = int(line[0])
                x = line[1]
                phi = CREATE_FEATURES(x)
                score = PREDICT_ONE(w, phi) * int(y)
                if score <= margin:
                    GETW(w, phi, y, c, last)

    with open('model_file.word','w') as model:
        for word,predict in sorted(w.items()):
            model.write('{} {}\n'.format(word,predict))

#test

with open("model_file.word") as text:
    w = defaultdict(float)
    for line in text:
        words = line.strip().split()
        w[words[0]] = float(words[1])


with open("../../data/titles-en-test.word") as text,open('my_answer.labeled', 'w') as answer:
    for line in text:
        phi = CREATE_FEATURES(line)
        if PREDICT_ONE(w, phi) >= 0:
            yy = 1
        else:
            yy = -1
        #yy = PREDICT_ONE(w, phi)
        answer.write(str(yy) + "\t" + str(line))
