from collections import defaultdict
import pickle
import random

def KeyMax(score):
    for k, v in sorted(score.items(), key = lambda x:x[1], reverse = True):
        return k

def ShiftReduceTrain(Q, H, weights):
    stack = [[0, 'ROOT', 'ROOT']]
    unproc = [0 for i in range(len(H))]
    score = dict()
    for head in H:
        if head == -1:
            continue
        unproc[int(head)] += 1
#    print(H)
    while len(Q) > 0 or len(stack) > 1:
        #print(Q)
        #print(stack)
        #print(H)
        #print(unproc)
        features = MakeFeatures(stack, Q)
        score['shift'] = PredictScore(weights['shift'], features)
        score['left'] = PredictScore(weights['left'], features)
        score['right'] = PredictScore(weights['right'], features)
        predict = KeyMax(score)


        if len(stack) < 2:
            correct = 'shift'
        elif H[stack[-1][0]] == stack[-2][0] and unproc[stack[-1][0]] == 0:
            correct = 'right'
        elif H[stack[-2][0]] == stack[-1][0] and unproc[stack[-2][0]] == 0:
            correct = 'left'
        else:
            correct = 'shift'
#            print(H[stack[-1][0]])
#            print(stack[-2][0])
#            print(unproc[stack[-1][0]])
#            print('1')

#        print(predict, correct)
        if predict != correct:
            UpdateWeights(weights, features, predict, correct)

        if correct == 'shift':
            stack.append(Q[0])
            del Q[0]
        elif correct == 'right':
            unproc[stack[-2][0]] -= 1
            del stack[-1]
        elif correct == 'left':
            unproc[stack[-1][0]] -= 1
            del stack[-2]

def MakeFeatures(stack, queue):
    features = defaultdict(lambda: 0)
    if len(stack) > 0 and len(queue) > 0:
        features['W-1' + stack[-1][1] + ',W0' + queue[0][1]] += 1
        features['W-1' + stack[-1][1] + ',P0' + queue[0][2]] += 1
        features['P-1' + stack[-1][2] + ',W0' + queue[0][1]] += 1
        features['P-1' + stack[-1][2] + ',P0' + queue[0][2]] += 1
    if len(stack) > 1:
        features['W-2' + stack[-2][1] + ',W-1' + stack[-1][1]] += 1
        features['W-2' + stack[-2][1] + ',P-1' + stack[-1][2]] += 1
        features['P-2' + stack[-2][2] + ',W-1' + stack[-1][1]] += 1
        features['P-2' + stack[-2][2] + ',P-1' + stack[-1][2]] += 1
    return features

def PredictScore(weight, features):
    score = 0
    for k, v in features.items():
        score += v * weight[k]
    return score

def UpdateWeights(weights, features, predict, correct):
    for k, v in features.items():
        weights[predict][k] -= 1 * v * l_rate
        weights[correct][k] += 1 * v * l_rate




if __name__ == '__main__':
    data = list()
    queue = list()
    heads = [-1]
    with open('../../data/mstparser-en-train.dep', 'r') as i_f:
        for line in i_f:
            if line == '\n':
                data.append([queue, heads])
                queue = list()
                heads = [-1]
            else:
                words = line.split()
                queue.append([int(words[0]), words[1], words[3]])
                heads.append(int(words[6]))
                print(words[3])

    weights = dict()
    weights['shift'] = defaultdict(int)
    weights['right'] = defaultdict(int)
    weights['left'] = defaultdict(int)
    l = 30
    l_rate = 1
    for i in range(l):
        random.shuffle(data)
        if i > 10:
            l_rate *= .7
        for Q, H in data:
            ShiftReduceTrain(Q, H, weights)

    with open('train_data.pkl', 'wb') as file_name:
        pickle.dump(weights, file_name)
