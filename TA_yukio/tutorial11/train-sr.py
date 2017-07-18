#python train-sr.py ../../data/mstparser-en-train.dep weights.txt 10

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

def make_features_default(stack, queue):
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
        score += weights[name] * value
    return score

def update_weights(weights, features, predict, correct):
    for name, value in features.items():
        weights[predict][name] -= value
        weights[correct][name] += value

def shift_reduce_train(queue, heads, weights):
    stack = [(0, "ROOT", "ROOT")]
    unproc = list()
    for i in range(len(heads)):
        unproc.append(heads.count(i))

    while len(queue) > 0 or len(stack) > 1:
        features = make_features(stack, queue)
        score_shift = predict_score(weights["shift"], features)
        score_left = predict_score(weights["left"], features)
        score_right = predict_score(weights["right"], features)

        if score_shift >= score_left and score_shift >= score_right and len(queue) > 0:
            predict = "shift"
        elif score_left >= score_right:
            predict = "left"
        else:
            predict = "right"

        if len(stack) < 2:
            correct = "shift"
        elif heads[stack[-1][0]] == stack[-2][0] and unproc[stack[-1][0]] == 0:
            correct = "right"
        elif heads[stack[-2][0]] == stack[-1][0] and unproc[stack[-2][0]] == 0:
            correct = "left"
        else:
            correct = "shift"

        if predict != correct:
            update_weights(weights, features, predict, correct)

        if correct == "shift":
            stack.append(queue.pop(0))
        elif correct == "left":
            unproc[stack[-1][0]] -= 1
            stack.pop(-2)
        else:
            unproc[stack[-2][0]] -= 1
            stack.pop(-1)

def train_sr(fin_path, weights_path, epoch):
    data = list()

    with open(fin_path, "r") as fin:
        queue = list()
        heads = [-1]
        for line in fin:
            if line == "\n":
                data.append((queue, heads))
                queue = list()
                heads = [-1]
            else:
                conll = line.strip("\n").split("\t")
                queue.append((int(conll[0]), conll[1], conll[3]))
                heads.append(int(conll[6]))
    
    weights = dict()
    weights["shift"] = defaultdict(lambda: 0)
    weights["left"] = defaultdict(lambda: 0)
    weights["right"] = defaultdict(lambda: 0)
    
    for i in range(epoch):
        random.seed(i)
        random.shuffle(data)
        for queue, heads in data:
            shift_reduce_train(queue[:], heads[:], weights)
    
    weights["shift"] = dict(weights["shift"])
    weights["left"] = dict(weights["left"])
    weights["right"] = dict(weights["right"])
    with open(weights_path, "wb") as fw:
        pickle.dump(weights, fw)

if __name__ == "__main__":
    train_sr(sys.argv[1], sys.argv[2], int(sys.argv[3]))
