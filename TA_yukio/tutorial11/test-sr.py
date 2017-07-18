#python test-sr.py ../../data/mstparser-en-test.dep weights.txt result.txt

from collections import defaultdict
import sys
import random
import pickle

def make_features(stack, queue):
    features = defaultdict(lambda: 0)
    if len(stack) > 1 and len(queue) > 0:
        features["W-2{},W-1{},W0{}".format(stack[-2][1], stack[-1][1], queue[0][1])] += 1
        features["W-2{},W-1{},P0{}".format(stack[-2][1], stack[-1][1], queue[0][2])] += 1
        features["W-2{},P-1{},W0{}".format(stack[-2][1], stack[-1][2], queue[0][1])] += 1
        features["W-2{},P-1{},P0{}".format(stack[-2][1], stack[-1][2], queue[0][2])] += 1
        features["P-2{},W-1{},W0{}".format(stack[-2][2], stack[-1][1], queue[0][1])] += 1
        features["P-2{},W-1{},P0{}".format(stack[-2][2], stack[-1][1], queue[0][2])] += 1
        features["P-2{},P-1{},W0{}".format(stack[-2][2], stack[-1][2], queue[0][1])] += 1
        features["P-2{},P-1{},P0{}".format(stack[-2][2], stack[-1][2], queue[0][2])] += 1
    if len(stack) > 0 and len(queue) > 0:
        features["W-1{},W0{}".format(stack[-1][1], queue[0][1])] += 1
        features["W-1{},P0{}".format(stack[-1][1], queue[0][2])] += 1
        features["P-1{},W0{}".format(stack[-1][2], queue[0][1])] += 1
        features["P-1{},P0{}".format(stack[-1][2], queue[0][2])] += 1
    if len(stack) > 1:
        features["W-2{},W-1{}".format(stack[-2][1], stack[-1][1])] += 1
        features["W-2{},P-1{}".format(stack[-2][1], stack[-1][2])] += 1
        features["P-2{},W-1{}".format(stack[-2][2], stack[-1][1])] += 1
        features["P-2{},P-1{}".format(stack[-2][2], stack[-1][2])] += 1
    if len(queue) > 0:
        features["W0{}".format(queue[0][1])] += 1
        features["P0{}".format(queue[0][2])] += 1
    if len(stack) > 0:
        features["W-1{}".format(stack[-1][1])] += 1
        features["P-1{}".format(stack[-1][2])] += 1
    if len(stack) > 1:
        features["W-2{}".format(stack[-2][1])] += 1
        features["P-2{}".format(stack[-2][2])] += 1
    return features

def make_features_deafult(stack, queue):
    features = defaultdict(lambda: 0)
    if len(stack) > 0 and len(queue) > 0:
        features["W-1{},W0{}".format(stack[-1][1], queue[0][1])] += 1
        features["W-1{},P0{}".format(stack[-1][1], queue[0][2])] += 1
        features["P-1{},W0{}".format(stack[-1][2], queue[0][1])] += 1
        features["P-1{},P0{}".format(stack[-1][2], queue[0][2])] += 1
    if len(stack) > 1:
        features["W-2{},W-1{}".format(stack[-2][1], stack[-1][1])] += 1
        features["W-2{},P-1{}".format(stack[-2][1], stack[-1][2])] += 1
        features["P-2{},W-1{}".format(stack[-2][2], stack[-1][1])] += 1
        features["P-2{},P-1{}".format(stack[-2][2], stack[-1][2])] += 1
    return features

def predict_score(weights, features):
    score = 0
    for name, value in features.items():
        if name in weights:
            score += weights[name] * value
    return score

def shift_reduce(queue, weights):
    heads = [-1] * (len(queue) + 1)
    stack = [(0, "ROOT", "ROOT")]

    while len(queue) > 0 or len(stack) > 1:
        features = make_features(stack, queue)
        score_shift = predict_score(weights["shift"], features)
        score_left = predict_score(weights["left"], features)
        score_right = predict_score(weights["right"], features)

        if len(stack) < 2 or (score_shift >= score_left and score_shift >= score_right and len(queue) > 0):
            stack.append(queue.pop(0))
        elif score_left >= score_right:
            heads[stack[-2][0]] = stack[-1][0]
            stack.pop(-2)
        else:
            heads[stack[-1][0]] = stack[-2][0]
            stack.pop(-1)
    return heads

def test_sr(fin_path, weights_path, fout_path):
    data = list()
    conll_data = list()

    with open(fin_path, "r") as fin:
        queue = list()
        conll_list = list()
        for line in fin:
            if line == "\n":
                data.append(queue)
                conll_data.append(conll_list)
                queue = list()
                conll_list = list()
            else:
                conll = line.strip("\n").split("\t")
                queue.append((int(conll[0]), conll[1], conll[3]))
                conll_list.append(conll)
    
    with open(weights_path, "rb") as fm:
        weights = pickle.load(fm)
   
    heads_list = list()
    for queue in data:
        heads_list.append(shift_reduce(queue, weights))

    with open(fout_path, "w") as fout:
        for heads, conll_list in zip(heads_list, conll_data):
            for head, conll in zip(heads[1:], conll_list):
                conll[6] = str(head)
                fout.write("{}\n".format("\t".join(conll)))
            fout.write("\n")

if __name__ == "__main__":
    test_sr(sys.argv[1], sys.argv[2], sys.argv[3])
