from collections import defaultdict,Counter
import pickle
from train_sr import MakeFeatures,PredictScore

def Shift_Reduce_Test(queue,weights):
    stack = [[0,'ROOT','ROOT']]
    heads = [-1] * (len(queue) + 1)
    while len(queue) > 0 or len(stack) > 1:
        features = MakeFeatures(stack,queue)
        s_shift = PredictScore(weights['shift'],features)
        s_left = PredictScore(weights['left'],features)
        s_right = PredictScore(weights['right'],features)
        if(s_shift >= s_left and s_shift >= s_right and len(queue) > 0) or len(stack) < 2:
            stack.append(queue.pop(0))
        elif s_left >= s_right:
            heads[stack[-2][0]] = stack[-1][0]
            stack.pop(-2)
        else:
            heads[stack[-1][0]] = stack[-2][0]
            stack.pop(-1)
    return heads

if __name__ == '__main__':
    with open('ShiftReduce_weights','rb') as t_f:
        weights = pickle.load(t_f)

    with open('../../data/mstparser-en-test.dep') as i_f:
        data = []
        queue = []
        heads = [-1]
        for line in i_f:
            if line == '\n':
                data.append([queue,heads])
                queue = []
                heads = [-1]
            else:
                ele = line.strip().split()
                queue.append([int(ele[0]),ele[1],ele[3]])
                heads.append(int(ele[6]))

    with open('../../data/mstparser-en-test.dep') as i_f,open('my_answer.txt','w') as o_f:            
        for queue in map(lambda x:x[0],data):
            heads = Shift_Reduce_Test(queue,weights)
            for i,line in enumerate(i_f):
                if line == '\n':
                    o_f.write('\n')
                    break
                else:
                    ans = '\t'.join(line.strip().split('\t')[0:6] + [str(heads[i+1])] + [line.strip().split('\t')[-1]])
                    o_f.write('{}\n'.format(ans))
