import pickle
from train_sr import *


def ShiftReduce(queue, weights):
    stack = [(0, "ROOT", "ROOT")]
    heads = [-1] * (len(queue) + 1)
    while len(queue) > 0 or len(stack) > 1:
        features = MakeFeatures(stack, queue)
        score = PredictScore(weights, features)
        # score_shift = PredictScore(weight_shift, features)
        # score_left = PredictScore(weight_left, features)
        # score_right = PredictScore(weight_right, features)
        if (score[0] > score[1] and score[0] > score[2] and len(queue) > 0) or len(stack) < 2:
            stack.append(queue.pop(0))
        elif score[1] > score[2] and score[1] > score[0]:
            heads[stack[-2][0]] = stack[-1][0]
            stack.pop(-2)
        else:
            heads[stack[-1][0]] = stack[-2][0]
            stack.pop(-1)
    return heads


if __name__ == "__main__":
        data = []
        queue = []
        heads = [-1]
        lines = []
        liness = []
        with open("../../data/mstparser-en-test.dep") as input_file:
            for line in input_file:
                line = line.strip()
                if line == '':
                    data.append(queue)
                    queue = list()
                    liness.append(lines)
                    lines = list()
                    # if line == "\n":
                    #     data.append(queue)
                    #     queue = 0
                else:
                    words = line.split("\t")
                    ID = words[0]
                    word = words[1]
                    pos = words[3]
                    head = words[6]
                    queue.append((int(ID), word, pos))
                    lines.append(line.split())
        with open("weight_sr.txt", 'rb') as weight_data:
            weights = pickle.load(weight_data)
        with open("answer_sr.txt", 'w') as answer_data:
            for i, queue in enumerate(data):
                heads = ShiftReduce(queue, weights)
                heads.pop(0)
                for j in range(len(heads)):
                    liness[i][j][6] = str(heads[j])
                    answer_data.write('{}\n'.format('\t'.join(liness[i][j])))
                answer_data.write('\n')
            # for queue in data:
            #     heads = ShiftReduce(queue, weights)
            #     output_file = open("head_file.txt", "w")
            #     output_file.write(heads)
