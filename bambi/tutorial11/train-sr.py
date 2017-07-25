from collections import  defaultdict
import pickle

RIGHT = "RIGHT"
LEFT = "LEFT"
SHIFT = "SHIFT"
def makeFeatures(stack,queue):
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

def predictScore(w, features):
    score = 0
    for k,v in features.items():
        score += w[k] * v
    return score

def updateWeights(w,features,label,correct):
    for k, v in features.items():
        if label == correct:
            w[k] += v
        else:
            w[k] -= v
    return w

def shiftReduceTrain(queue,heads,weight_shift,weight_left,weight_right):
    stack = [(0,"ROOT","ROOT")]
    unproc = []
    for i in range(len(heads)):
        unproc.append(heads.count(i))

    while len(queue) > 0 or len(stack) > 1:
        features = makeFeatures(stack,queue)

        score_shift = predictScore(weight_shift,features)
        score_left = predictScore(weight_left,features)
        score_right = predictScore(weight_right,features)

        arg_max = max([score_shift, score_left, score_right])
        predict = ""
        correct = ""
        if (arg_max == score_shift and len(queue) > 0) or len(stack) < 2:
            predict = SHIFT
        elif arg_max == score_left:
            predict = LEFT
        else:
            predict = RIGHT

        if len(stack) < 2:
            correct = SHIFT
        elif heads[stack[-1][0]] == stack[-2][0] and unproc[stack[-1][0]] == 0:
            correct = RIGHT
        elif heads[stack[-2][0]] == stack[-1][0] and unproc[stack[-2][0]] == 0:
            correct = LEFT
        else:
            correct = SHIFT

        if predict != correct:
            weight_shift = updateWeights(weight_shift,features,SHIFT,correct)
            weight_right = updateWeights(weight_right,features,RIGHT,correct)
            weight_left = updateWeights(weight_left,features,LEFT,correct)

        if correct == SHIFT:
            stack.append(queue.pop(0))
        elif correct == LEFT:
            unproc[int(stack[-1][0])] -= 1
            stack.pop(-2)
        elif correct == RIGHT:
            unproc[int(stack[-2][0])] -= 1
            stack.pop(-1)


data = []
queue = []
heads = [-1]

file = "../data/mstparser-en-train.dep"
for line in open(file):
    if line == '\n':
        data.append((queue,heads))
        queue = []
        heads = [-1]
    else:
        items = line.split("\t")
        word_id = int(items[0])
        word = items[1]
        pos = items[3]
        head = int(items[6])
        queue.append((word_id,word,pos))
        heads.append(head)

weight_shift = defaultdict(int)
weight_right = defaultdict(int)
weight_left = defaultdict(int)

l = 5

for i in range(l):
    print("round {}".format(i))
    for queue, heads in data:
        shiftReduceTrain(queue,heads,weight_shift,weight_left,weight_right)


pickle.dump(weight_shift,open("weight_shift.pickle","wb"))
pickle.dump(weight_right,open("weight_right.pickle","wb"))
pickle.dump(weight_left,open("weight_left.pickle","wb"))
