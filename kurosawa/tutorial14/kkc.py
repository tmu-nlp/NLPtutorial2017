import pickle
from collections import defaultdict
import math

def forward_step(line,lm,tm):
    lambda_lmprob = 0.000001
    lambda_tmprob = 0.00001
    edge = defaultdict(dict)
    score = defaultdict(lambda:defaultdict(int))
    edge[0]['<s>'] = 'NULL'
    score[0]['<s>'] = 0
    for end in range(1,len(line)+1):
        score[end] = {}
        edge[end] = {}
        for begin in range(end):
            pron = line[begin:end]
            if pron not in tm and len(pron) == 1:
                my_tm = {pron:0}
            else:
                my_tm = tm[pron]
#            print(pron)
            for curr_word,tm_prob in my_tm.items():
                for prev_word,prev_score in score[begin].items():
                    try:
                        lm_prob = lambda_lmprob + (1-lambda_lmprob)*lm['{} {}'.format(prev_word,curr_word)]
                    except:
                        lm_prob = lambda_lmprob
                    if tm_prob == 0:
                        tm_prob = lambda_tmprob
                    else:
                        tm_prob = lambda_tmprob + (1-lambda_tmprob)*tm_prob
#                print(curr_word)
                    curr_score = prev_score - math.log2(tm_prob*lm_prob)
                    if curr_word in score[end]:
                        if curr_score < score[end][curr_word]:
                            score[end][curr_word] = curr_score
                            edge[end][curr_word] = (begin,prev_word)
                    else:
                        score[end][curr_word] = curr_score
                        edge[end][curr_word] = (begin,prev_word)

    for prev_word,prev_score in score[end].items():
        if '{} </s>'.format(prev_word) in lm:
            curr_score = prev_score - math.log2(lambda_lmprob + (1-lambda_lmprob)*lm['{} </s>'.format(prev_word)])
        else:
            curr_score = prev_score - math.log2(lambda_lmprob)
        if '</s>' in score[end+1]:
            if curr_score < score[end+1]['</s>']:
                score[end+1]['</s>'] = curr_score
                edge[end+1]['</s>'] = (end,prev_word)
        else:
            score[end+1]['</s>'] = curr_score
            edge[end+1]['</s>'] = (end,prev_word)

    tags = []
    next_edge = edge[end+1]['</s>']
    while next_edge != 'NULL':
        position,tag = next_edge
        tags.append(tag)
        next_edge = edge[position][tag]
    tags.pop()
    tags.reverse()
    return tags

def backward_step(edge):
    print(edge) 
    tags = []

if __name__ =='__main__':
    lm = pickle.load(open('lm.txt','rb'))
    tm = pickle.load(open('tm.txt','rb'))

    with open('../../data/wiki-ja-test.pron') as test_file,open('output.txt','w') as answer:
        for line in test_file:
            tags = forward_step(line.strip(),lm,tm)
            answer.write('{}\n'.format(''.join(tags)))
