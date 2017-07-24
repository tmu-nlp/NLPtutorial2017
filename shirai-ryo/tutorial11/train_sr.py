from collections import defaultdict
import pickle

def ShiftReduceTrain(queue, heads, weights):
    stack = [(0, "ROOT", "ROOT")]
    unproc = []
    for i in range(len(heads)):
        unproc.append(heads[i])
    print(queue)
    while len(queue) > 0 or len(stack) > 1:
        features = MakeFeatures(stack, queue)
        score = PredictScore(weights, features)
        # score_shift = PredictScore(weight_shift, features)
        # score_left = PredictScore(weight_left, features)
        # score_right = PredictScore(weight_right, features)
        if (score[0] > score[1] and score[0] > score[2] and len(queue) > 0) or len(stack) < 2:
            predict = "shift"
        elif score[1] > score[2] and score[1] > score[0]:
            predict = "left"
        else:
            predict = "right"

        if len(stack) < 2:
            correct = "shift"
        elif heads[stack[-1][0]] == stack[-2][0] and unproc[stack[-1][0]] == 0:
            correct = "right"
        else:
            correct = "left"

        # print(predict)
        # print("---")
        # print(correct)


        if predict != correct:
            UpdateWeights(weights, features, predict, correct)
            #print("ウェイ")

        # pop()はリストの中の指定したインデックスの要素を削除する

        if correct == "shift":
            stack.append(queue.pop(0))
        elif correct == "left":
            unproc[stack[-1][0]] -= 1
            stack.pop(-2)
        elif correct == "right":
            unproc[stack[-2][0]] -= 1
            stack.pop(-1)


def MakeFeatures(stack, queue):
    features = defaultdict(int)
    if len(stack) > 0 and len(queue) > 0:
        features["W-1" + stack[-1][1] + ",W0" + queue[0][1]] += 1
        features["W-1" + stack[-1][1] + ",P0" + queue[0][2]] += 1
        features["P-1" + stack[-1][2] + ",W0" + queue[0][1]] += 1
        features["P-1" + stack[-1][2] + ",P0" + queue[0][2]] += 1
    if len(stack) > 1:
        features["W-2" + stack[-2][1] + ",W-1" + stack[-1][1]] += 1
        features["W-2" + stack[-2][1] + ",P-1" + stack[-1][2]] += 1
        features["P-2" + stack[-2][2] + ",W-1" + stack[-1][1]] += 1
        features["P-2" + stack[-2][2] + ",P-1" + stack[-1][2]] += 1
    return features


def UpdateWeights(weights, features, predict, correct):
    for name, value in features.items():
        if predict == "shift":
            weights[0][name] -= value
        elif predict == "left":
            weights[1][name] -= value
        else:
            weights[2][name] -= value
        if correct == "shift":
            weights[0][name] += value
        elif correct == "left":
            weights[1][name] += value
        else:
            weights[2][name] += value

def PredictScore(weights, features):
    score = [0,0,0]
    for name, value in features.items():
        for i in range(3):
            #print(type(name), type(value))
            #print(type(weights[i][name]))
            score[i] += value * weights[i][name]
    return score




# Training
if __name__ == "__main__":
    with open("../../data/mstparser-en-train.dep") as input_file:
        epoch = 10
        data = []
        queue = []
        heads = [-1]
        for line in input_file:
            if line == "\n":
                data.append((queue, heads))
                queue = []
                heads = [-1]
            else:
                words = line.strip().split("\t")
                # print(words)
                ID = words[0]
                word = words[1]
                pos = words[3]
                head = words[6]
                queue.append((int(ID), word, pos))
                heads.append(int(head))
        weight_shift = defaultdict(int)
        weight_left = defaultdict(int)
        weight_right = defaultdict(int)
        weights = [weight_shift, weight_left, weight_right]

        for epoch in range(epoch):
            for queue, heads in data:
                ShiftReduceTrain(queue[:], heads[:], weights)
        with open("weight_sr.txt", "wb") as output_file:
            pickle.dump(weights, output_file)
