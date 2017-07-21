
# coding: utf-8

# In[3]:


from collections import defaultdict
import pickle


# In[2]:


test_file="../../data/mstparser-en-test.dep"


# In[12]:


data,queue=[],[]
with open("heads.txt","w",encoding="utf-8"):
    pass

for line in open(test_file,"r",encoding="utf-8"):
    line=line[:-1]
    

    if len(line) == 0:
        data.append(queue)
        queue = []
    else:
        ID, word, base, POS, POS2, _, head, type_ = line.split("\t")
        head = int(head)-1
        ID = int(ID)-1
        queue.append((ID, word, POS))

weight_shift, weight_left, weight_right=pickle.load(open("./weights.pkl","rb"))

for que in data:
    heads=ShiftReduce(que,weight_shift, weight_left, weight_right)
    print(heads,file=open("heads.txt","a",encoding="utf-8"))


# In[11]:


def ShiftReduce(queue,weight_shift, weight_left, weight_right):
    stack=[(0,"ROOT","ROOT")]
    heads=[-1]*(len(queue)+1)
    while len(queue)>0 or len(stack)>1:
        features=MakeFeatures(stack,queue)
        
        score_shift = PredictScore(weight_shift, features)
        score_left = PredictScore(weight_left, features)
        score_right = PredictScore(weight_right, features)
        
        if (score_shift > max(score_left, score_right) and
                len(queue) > 0) or len(stack) < 2:
            stack.append(queue.pop(0))
        elif score_left > max(score_shift, score_right):
            heads[stack[-2][0]]=stack[-1][0]
            stack.pop(-2)
        else:
            heads[stack[-1][0]]=stack[-2][0]
            stack.pop(-1)
    return heads


# In[4]:


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


# In[5]:


def PredictScore(weight,features):
    score=0
    for feat in features:
        score+=weight[feat]*features[feat]
    return score

