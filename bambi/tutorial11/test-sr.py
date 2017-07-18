import pickle
from collections import  defaultdict

def predictScore(w, features):
    score = 0
    for k,v in features.items():
        score += w[k] * v
    return score

def makeFeatures(stack, queue):
    features = defaultdict(int)
    if len(queue) > 0 and len(stack) >0:
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

def shiftReduce(queue,weight_shift,weight_left,weight_right):
    stack = [(0,"ROOT","ROOT")]
    heads = defaultdict(int)

    while len(queue) > 0 or len(stack) > 1:
        features = makeFeatures(stack,queue)
        score_shift = predictScore(weight_shift,features)
        score_left = predictScore(weight_left,features)
        score_right = predictScore(weight_right,features)
        arg_max = max([score_shift, score_left, score_right])
        if (arg_max == score_shift and len(queue)) > 0 or len(stack) < 2:
            stack.append(queue.pop(0))
        elif arg_max == score_left:
            heads[int(stack[-2][0])] = int(stack[-1][0])
            stack.pop(-2)
        else:
            heads[int(stack[-1][0])] = int(stack[-2][0])
            stack.pop(-1)
    return heads

data = []
queue = []
file = "../data/mstparser-en-test.dep"
weight_shift = pickle.load(open("weight_shift.pickle","rb"))
weight_right = pickle.load(open("weight_right.pickle","rb"))
weight_left = pickle.load(open("weight_left.pickle","rb"))


for line in open(file, "r"):
    line = line.strip("\n")
    items = line.split("\t")
    if len(line) == 0:
        data.append(list(queue))
        queue = []
    else:
        word_id = items[0]
        word = items[1]
        pos = items[3]
        word_type = items[-1]
        queue.append((word_id,word,pos,word_type))

data.append(queue) # add final queue

with open("answer.dep","w") as output:
    for queue in data:
        copy_queue = list(queue)
        heads = shiftReduce(queue,weight_shift,weight_left,weight_right)
        for v in copy_queue:
            word_id = int(v[0])
            word = v[1]
            word_pos = v[-2]
            word_type = v[-1]
            print("{}\t{}\t{}\t{}\t{}\t_\t{}\t{}".format(word_id,word,word,word_pos,word_pos,heads[word_id],word_type),file=output)
        print(file=output)# add \n to match comparing file

print("finished")
