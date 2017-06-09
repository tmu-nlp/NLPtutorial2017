#python test-perceptron.py ../../data/titles-en-test.word model.txt my_answer.word
#../../script/grade-prediction.py ../../data/titles-en-test.labeled my_answer.word

from collections import defaultdict
import sys
import re

def PREDICT_ONE(w, phi):
    score = 0
    for name, value in phi.items():
        if name in w:
            score += value * w[name]
    if score >= 0:
        return 1
    else:
        return -1

def CREATE_FEATURES(x):
    phi = defaultdict(lambda: 0)
    x = x.lower()
    x = re.sub(r'''["'-\/–#・(),.!?;:{}^~－（）ー〜\[\]<>＜＞［］]''', "", x)
    words = x.split()
    for word1, word2, word3 in zip(words, words[1:], words[2:]):
        phi["UNI:{}".format(word1)] += 1
        phi["BI:{} {}".format(word1, word2)] += 1
        phi["TRI:{} {} {}".format(word1, word2, word3)] += 1
    if len(words) > 0:
        phi["UNI:{}".format(words[-1])] += 1
    if len(words) > 1:
        phi["UNI:{}".format(words[-2])] += 1
        phi["BI:{} {}".format(words[-2], words[-1])] += 1
    return phi

def PREDICT_ALL(model_file, input_file):
    w = defaultdict(lambda: 0)
    with open(input_file, "r") as fin, open(model_file, "r") as fm:
        for line in fm:
            name, value = line.split("\t")
            w[name] = int(value)

        for x in fin:
            phi = CREATE_FEATURES(x)
            y = PREDICT_ONE(w, phi)
            yield y

def test_perceptron(fin_path, model_path, fout_path):
    with open(fout_path, "w") as fout:
        for y in PREDICT_ALL(model_path, fin_path):
            fout.write("{}\n".format(y))

if __name__ == "__main__":
    test_perceptron(sys.argv[1], sys.argv[2], sys.argv[3])
