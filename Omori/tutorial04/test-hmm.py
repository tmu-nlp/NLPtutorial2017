from collections import defaultdict
import math
import sys

def load_model(model_file):
    transition = defaultdict(float)
    emission = defaultdict(float)
    possible_tags = defaultdict(int)
    
    with open(model_file, 'r') as f:
        for line in f:
            _type, w1, w2, prob = line.strip().split()
            possible_tags[w1] = 1
            if _type == 'T':
                transition[w1+' '+w2] = float(prob)
            else:
                emission[w1+' '+w2] = float(prob)

    return transition, emission, possible_tags

def main(input_file):
    # for unigram model
    lambda_1 = 0.95
    lambda_unk = 1 - lambda_1
    V = 1000000

    with open(input_file, 'r') as f:
        for line in f:
            # forward step
            best_edge = defaultdict(str)
            best_score = defaultdict(float)
            best_edge["0 <s>"] = 'NULL'
            best_score["0 <s>"] = 0.0
            line = line.strip().split() # remove new line
            length = len(line)
            for i in range(0,length):
                for prev in possible_tags.keys():
                    for _next in possible_tags.keys():
                        temp1 = '{} {}'.format(prev, _next)
                        temp2 = '{} {}'.format(_next, line[i])
                        temp3 = '{} {}'.format(i, prev)
                        temp4 = '{} {}'.format(i+1, _next)
                        if temp1 in transition and temp3 in best_score:
                            prob_T = transition[temp1]
                            prob_E = lambda_1 * emission[temp2] + lambda_unk / V
                            score = best_score[temp3] - math.log(prob_T, 2) - math.log(prob_E, 2) 
                            #if best_score[temp4] > score or temp4 not in best_score:  # miss imitation code
                            if temp4 not in best_score or best_score[temp4] > score:  # miss imitation code
                                best_score[temp4] = score
                                best_edge[temp4] = '{} {}'.format(i, prev)
            for prev in possible_tags.keys():
                temp5 = '{} </s>'.format(length+1)
                if prev+' </s>' in transition and '{} {}'.format(length, prev) in best_score:
                    prob_T_end = transition[prev+' </s>']    
                    score = best_score['{} {}'.format(length, prev)] - math.log(prob_T_end, 2)
                    #if best_score[temp5] > score or temp5 not in best_score:  # miss imitation code
                    if temp5 not in best_score or best_score[temp5] > score:  # miss imitation code
                        best_score[temp5] = score
                        best_edge[temp5] = '{} {}'.format(length, prev)
            
            # back step
            tags = list()
            next_edge = best_edge['{} </s>'.format(length+1)]  # next_edge='length prev'
            while next_edge != '0 <s>':
                position, tag =  next_edge.split()
                tags.append(tag)
                next_edge = best_edge[next_edge]
            tags.reverse()
            print(' '.join(tags))

if __name__ == "__main__":
    transition, emission, possible_tags = load_model(sys.argv[1])
    main(sys.argv[2])

