import pickle
import train_sr

def ShiftReduce(queue):
    stack = [(0, 'ROOT', 'ROOT')]
    heads = [-1] * (len(queue) + 1) 
    while len(queue) > 0 or len(stack) > 1:
        features = train_sr.MakeFeatures(stack,queue)
        score_shift = train_sr.PredictScore(weight_shift, features)
        score_left = train_sr.PredictScore(weight_left, features)
        score_right = train_sr.PredictScore(weight_right, features)
        if (score_shift >= score_left and score_shift >= score_right and len(queue) > 0) or len(stack) < 2:
            stack.append(queue.pop(0))
        elif score_left >= score_right:
            heads[stack[-2][0]] = stack[-1][0]
            stack.pop(-2)
        else:
            heads[stack[-1][0]] = stack[-2][0]
            stack.pop(-1)
    return heads    
    

if __name__ == '__main__':
    line_list = list()
    word = list()
    data = list()
    queue = list()
    with open('../../data/mstparser-en-test.dep') as i_f:
        for line in i_f:
            if line.strip() == '':
                data.append(queue)
                line_list.append(word)
                queue = list()
                word = list()
            else:
                elements = line.strip().split()
                word.append(elements)
                queue.append((int(elements[0]), elements[1], elements[3]))
    with open('weights.dump', 'rb') as w_f:
        weight_shift, weight_left, weight_right = pickle.load(w_f)
    with open('my_answer.txt', 'w') as o_f:
        for queue, line in zip(data, line_list):
            heads = ShiftReduce(queue)
            heads.pop(0)
            for head, word in zip(heads, line):
                word[6] = str(head)
                o_f.write('\t'.join(word)+'\n')
                #o_f.write('{}\n'.format(head))
            o_f.write('\n')
