from collections import defaultdict
import random
import copy
import pickle

def ShiftReduceTrain(queue, heads, count, lam=.01):
    stack = [(0, 'ROOT', 'ROOT')]
    unproc = list()
    for i in range(len(heads)):
        unproc.append(heads.count(i))
    while len(queue) > 0 or len(stack) > 1:
        features = MakeFeatures(stack, queue)
        score_shift = PredictScore(weight_shift, features)
        score_left = PredictScore(weight_left, features)
        score_right = PredictScore(weight_right, features)
        #scoreを元にshift left rightを選択する
        if (score_shift >= score_left and score_shift >= score_right and len(queue) > 0) or len(stack) < 2:
            predict = 'shift'
        elif score_left >= score_right:
            predict = 'left'
        else:
            predict = 'right'
        #与えられているheadからshift left rightを逆算
        if len(stack) < 2:
            correct = 'shift'
        elif heads[int(stack[-1][0])] is stack[-2][0] and unproc[int(stack[-1][0])] is 0:
            correct = 'right'
        elif heads[int(stack[-2][0])] is stack[-1][0] and unproc[int(stack[-2][0])] is 0:
            correct = 'left'
        else:
            correct = 'shift'
        if predict != correct:
            count += 1
            UpdateWeights(features, predict, correct, lam)

        if correct == 'shift':
            stack.append(queue.pop(0))
        elif correct == 'left':
            unproc[int(stack[-1][0])] -= 1
            stack.pop(-2)
        elif correct == 'right':
            unproc[int(stack[-2][0])] -= 1
            stack.pop(-1)
    return count
    

def MakeFeatures(stack, queue):
    features = defaultdict(int)
    if len(stack) > 0 and len(queue) > 0:
        features['W-1{},W0{}'.format(stack[-1][1],queue[0][1])] += 1
        features['W-1{},P0{}'.format(stack[-1][1],queue[0][2])] += 1
        features['P-1{},W0{}'.format(stack[-1][2],queue[0][1])] += 1
        features['P-1{},P0{}'.format(stack[-1][2],queue[0][2])] += 1
    if len(stack) > 1:
        features['W-2{},W-1{}'.format(stack[-2][1],stack[-1][1])] += 1
        features['W-2{},P-1{}'.format(stack[-2][1],stack[-1][2])] += 1
        features['P-2{},W-1{}'.format(stack[-2][2],stack[-1][1])] += 1
        features['P-2{},P-1{}'.format(stack[-2][2],stack[-1][2])] += 1
    return features

def UpdateWeights(features, predict, correct, lam):
    if predict is 'shift':
        for key, value in features.items():
            weight_shift[key] -= value*lam
            if correct is 'left':
                weight_left[key] += value*lam
            else:
                weight_right[key] += value*lam
    elif predict is 'left':
        for key, value in features.items():
            weight_left[key] -= value*lam
            if correct is 'shift':
                weight_shift[key] += value*lam
            else:
                weight_right[key] += value*lam
    else:
        for key, value in features.items():
            weight_right[key] -= value*lam
            if correct is 'shift':
                weight_shift[key] += value*lam
            else:
                weight_left[key] += value*lam

def PredictScore(weight, features):
    score = 0
    for key, value in features.items():
        if key in weight:
            score += weight[key] * value
    return score
    
"""
weightsを辞書で作らなかったのは謎
"""
if __name__ == '__main__':
    data = list()
    queue = list()
    heads = [-1]
    l = 10
    weight_shift = defaultdict(int)
    weight_left = defaultdict(int)
    weight_right = defaultdict(int)
    lam = 1
    with open('../../data/mstparser-en-train.dep') as i_f:
        for line in i_f:
            if line.strip() == '':
                data.append((queue, heads))
                queue = list()
                heads = [-1]
            else:
                elements = line.strip().split()
                queue.append((int(elements[0]), elements[1], elements[3]))
                heads.append(int(elements[6]))
    for _ in range(l):
        data_util = copy.deepcopy(data)
        random.shuffle(data_util)
        count = 0
        print (_)
        for q, h in data_util:
            count = ShiftReduceTrain(q, h, count, lam)
        print ('count:{}'.format(count))
    with open('weights.dump', 'wb') as o_f:
        pickle.dump((dict(weight_shift), dict(weight_left), dict(weight_right)), o_f)
