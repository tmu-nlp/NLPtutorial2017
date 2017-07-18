from collections import defaultdict,Counter
import pickle

def Shift_Reduce_Train(queue,heads,weights):
    stack = [[0,'ROOT','ROOT']]
    unproc = []
    for i in range(len(heads)):
        unproc.append(heads.count(i))
#        print(unproc)
    while len(queue) > 0 or len(stack) > 1:
        features = MakeFeatures(stack,queue)
    #    print(features)
    # 予測スコアの計算
        s_shift = PredictScore(weights['shift'],features) #predictscore?
        s_left = PredictScore(weights['left'],features)
        s_right = PredictScore(weights['right'],features)
    # 予測での分岐
        if(s_shift >= s_left and s_shift >= s_right and len(queue) > 0) or len(stack) < 2:
            predict = 'shift'
        elif s_left >= s_right:
            predict = 'left'
        else:
            predict = 'right'
    # 正解の導出
        if len(stack) < 2:
            correct = 'shift'
        elif heads[stack[-1][0]] == stack[-2][0] and unproc[stack[-1][0]] == 0:
        #elif heads[stack[-1][0]] is stack[-2][0] and unproc[stack[-1][0]] is 0:
            correct = 'right'
        elif heads[stack[-2][0]] == stack[-1][0] and unproc[stack[-2][0]] == 0:
        #elif heads[stack[-2][0]] is stack[-1][0] and unproc[stack[-2][0]] is 0:
            correct = 'left'
        else:
            correct = 'shift'
    # 条件で重み更新
        if predict != correct:
            weights = Update_w(weights,feature,predict,correct)

        if correct == 'shift':
            stack.append(queue.pop(0))
        elif correct == 'left':
            unproc[stack[-1][0]] -= 1
            stack.pop(-2)
        elif correct == 'right':
            unproc[stack[-2][0]] -= 1
            stack.pop(-1)

def Update_w(weights,feature,predict,correct):
    for key,value in features.items():
        weights[predict][key] -= features[key]
        weights[correct][key] += features[key]
    return weights

def PredictScore(weights,features):
    score = 0
    for key,value in features.items():
        if key in features.keys():
            score += value * weights[key]
    return score


def MakeFeatures(stack, queue):
    features = defaultdict(int)
    if len(queue) > 0 and len(stack) >0:
        features['W-1' + stack[-1][1] + ',W0' + queue[0][1]] += 1
        features['W-1' + stack[-1][1] + ',P0' + queue[0][2]] += 1
        features['P-1' + stack[-1][2] + ',W0' + queue[0][1]] += 1
        features['P-1' + stack[-1][2] + ',P0' + queue[0][2]] += 1
    if len(stack) > 1:
        features['W-2' + stack[-2][1] + ',W-1' + stack[-1][1]] += 1
        features['W-2' + stack[-2][1] + ',P-1' + stack[-1][2]] += 1
        features['P-2' + stack[-2][2] + ',W-1' + stack[-1][1]] += 1
        features['P-2' + stack[-2][2] + ',P-1' + stack[-1][2]] += 1
    #    print(features)
    return features


if __name__ == '__main__':
    with open('../../data/mstparser-en-train.dep') as i_f:
        data = []
        queue = []
        heads = [-1]
        epoch = int(input('> epoch:'))
        for line in i_f:
            if line == '\n':
                data.append([queue,heads])
                queue = []
                heads = [-1]
            else:
                ele = line.strip().split()
                queue.append([int(ele[0]),ele[1],ele[3]])
                heads.append(int(ele[6]))

        weights = dict()
        weights['shift'] = defaultdict(int)
        weights['left'] = defaultdict(int)
        weights['right'] = defaultdict(int)
        # weightsがないので三種のdictをweights(dict?)に突っ込みたい
        # この仕様であってるかは謎
        for i in range(epoch):
            for q,h in data:
    #            print(q,h)
                Shift_Reduce_Train(queue,heads,weights)
        with open('ShiftReduce_weights','wb') as o_f:
            pickle.dump(weights,o_f)
