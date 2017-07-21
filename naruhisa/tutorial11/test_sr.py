from collections import defaultdict
from train_sr import MakeFeatures, PredictScore, KeyMax
import pickle


def MakeOutput(origin, heads):
    for i in range(len(origin)):
        origin[i][6] =  heads[i]
    return origin

def ShiftReduce(queue, weights):
    score = dict()
    stack = [[0, 'ROOT', 'ROOT']]
    heads = [-1] * (len(queue) + 1)
    while len(stack) > 1 or len(queue) > 0:
        features = MakeFeatures(stack, queue)
        score['shift'] = PredictScore(weights['shift'], features)
        score['left'] = PredictScore(weights['left'], features)
        score['right'] = PredictScore(weights['right'], features)
        predict = KeyMax(score)
        if (predict == 'shift' and len(queue) > 0) or len(stack) < 2:
            stack.append(queue[0])
            del queue[0]
        elif predict == 'left':
            heads[stack[-2][0]] = stack[-1][0]
            del stack[-2]
        else:
            heads[stack[-1][0]] = stack[-2][0]
            del stack[-1]
    del heads[0]
    return heads

if __name__ == '__main__':
    data = list()
    queue = list()
    origin = list()
    o_data = list()
    with open('../../data/mstparser-en-test.dep') as i_f:
        for line in i_f:
            if line == '\n':
                queue.append(data)
                origin.append(o_data)
                data = list()
                o_data = list()
            else:
                words = line.split()
                data.append([int(words[0]), words[1], words[3]])
                words = str(line).split()
                o_data.append(words)
    with open('train_data.pkl', 'rb') as t_f:
        weights = pickle.load(t_f)

    with open('result.txt', 'w') as o_f:
        for Q, O in zip(queue, origin):
            heads = ShiftReduce(Q, weights)
            outputs = MakeOutput(O, heads)
            for output in outputs:
                for i in range(len(output)):
                    output[i] = str(output[i])
                o_f.write('\t'.join(output) + '\n')
            o_f.write('\n')
