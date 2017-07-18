
# coding: utf-8

# In[47]:


from collections import defaultdict
import pickle
from pprint import pprint


# In[48]:


training_file="../../data/mstparser-en-train.dep"
num_iter=100


# In[ ]:


data, queue, heads = [], [], [-1]
for idx, line in enumerate(open(training_file, "r", encoding="utf-8")):
    line = line[:-1]

    if len(line) == 0:
        data.append((queue, heads))
        heads, queue = [], []
    else:
        ID, word, base, POS, POS2, _, head, type_ = line.split("\t")
        head = int(head)-1
        ID = int(ID)-1
        queue.append((ID, word, POS))
        heads.append(head)

weight_shift = defaultdict(float)
weight_left = defaultdict(float)
weight_right = defaultdict(float)

for i in range(num_iter):
    if i % 10 == 0:
        print("training iteration {}".format(i))
    for queue, heads in data:
        
        weight_shift, weight_left, weight_right = ShiftReduceTrain(
            queue, heads, weight_shift, weight_left, weight_right)
        

pickle.dump((weight_shift, weight_left, weight_right),
            open("weights.pkl", "wb"))


# In[108]:


def ShiftReduceTrain(queue, heads, weight_shift, weight_left, weight_right):
    stack = [(0, "ROOT", "ROOT")]
    unproc = []
    for i in range(len(heads)):
        unproc.append(heads.count(i))

    while len(queue) > 0 or len(stack) >1:
        features = MakeFeatures(stack, queue)

        score_shift = PredictScore(weight_shift, features)
        score_left = PredictScore(weight_left, features)
        score_right = PredictScore(weight_right, features)

        if (score_shift > max(score_left, score_right) and
                len(queue) > 0) or len(stack) < 2:
            predict = "shift"
        elif score_left > max(score_shift, score_right):
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
            weight_shift, weight_left, weight_right = UpdateWeights(
                weight_shift, weight_left, weight_right, features, predict,
                correct)
        
        if correct == "shift":
            try:
                stack.append(queue.pop(0))
            except IndexError:
                print(stack)
                input()
        elif correct == "left":
            unproc[stack[-1][0]] -= 1
            stack.pop(-2)
        elif correct == "right":
            unproc[stack[-2][0]] -= 1
            stack.pop(-1)
    return weight_shift, weight_left, weight_right


# In[77]:


def MakeFeatures(stack, queue):
    features=defaultdict(int)
    
    if len(stack)>0 and len(queue)>0:
        features[("W-1",stack[-1][1],"W0",queue[0][1])]+=1
        features[("W-1",stack[-1][1],"P0",queue[0][2])]+=1
        features[("P-1",stack[-1][2],"W0",queue[0][1])]+=1
        features[("P-1",stack[-1][2],"P0",queue[0][2])]+=1
    
    if len(stack)>1:
        try:
            features[("W-2",stack[-2][1],"W-1",queue[-1][1])]+=1
            features[("W-2",stack[-2][1],"P-1",queue[-1][2])]+=1
            features[("P-2",stack[-2][2],"W-1",queue[-1][1])]+=1
            features[("P-2",stack[-2][2],"P-1",queue[-1][2])]+=1
        except:
            pass
    return features


# In[74]:


def PredictScore(weight,features):
    score=0
    for feat in features:
        score+=weight[feat]*features[feat]
    return score
    


# In[75]:


def UpdateWeights(weight_shift, weight_left, weight_right,
                  features,predict, correct):
    if predict=="shift" and correct=="left":
        for feat in features:
            weight_shift[feat]-=features[feat]
            weight_left[feat]+=features[feat]
    if predict=="shift" and correct=="right":
        for feat in features:
            weight_shift[feat]-=features[feat]
            weight_right[feat]+=features[feat]
    if predict=="right" and correct=="left":
        for feat in features:
            weight_right[feat]-=features[feat]
            weight_left[feat]+=features[feat]
    if predict=="right" and correct=="shift":
        for feat in features:
            weight_right[feat]-=features[feat]
            weight_shift[feat]+=features[feat]
    if predict=="left" and correct=="right":
        for feat in features:
            weight_left[feat]-=features[feat]
            weight_right[feat]+=features[feat]
    if predict=="left" and correct=="shift":
        for feat in features:
            weight_left[feat]-=features[feat]
            weight_shift[feat]+=features[feat]
    return weight_shift,weight_left,weight_right

