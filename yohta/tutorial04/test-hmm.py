from collections import defaultdict
import math

transition = defaultdict(int)
emission = defaultdict(int)
possible_tags = defaultdict(int)
"""
model-hmm
[0]=type
[1]=context
[2]=word
[3]=prob
[0] [1] [2] [3]
"""
with open('model-hmm.txt','r') as f:
    for line in f:
        tcwp = line.split(' ')
        typ = tcwp[0]
        context = tcwp[1]
        word = tcwp[2]
        prob = tcwp[3]
        possible_tags[context] = 1
        if typ == 'T':
            transition[context + ' ' + word] = prob
        else:
            emission[context + ' ' + word] = prob

best_score = defaultdict(lambda: 0)
best_edge = defaultdict(lambda: 0)


with open('../../data/wiki-en-test.norm','r') as t_f, open('my_answer.pos','w') as a_f:
    for line in t_f:
        words = line.split()
#        print(words)
        best_score['0 <s>'] = 0
        best_edge['0 <s>'] = 'NULL'
        for i in range(len(words)):
            for prev in possible_tags.keys():
                for nex in possible_tags.keys():
                    if best_score[i + ' ' + prev] in best_score and transition[prev + ' ' + nex] in transition:
                        score = best_score[i + ' ' + prev] - math.log2(transition[prev + ' ' + nex])
                        P = lambda_unk/V
                        if nex + ' ' + words[i] in emission:
                            P += lambda_1 * emission[nex + ' ' + words[i]]
                        score += -math.log2(P)
                        if not best_score[i+1 + ' ' + nex] in best_score or best_score[i+1 + ' ' + nex] < score:
                            best_score[i+1 + ' ' + nex] = score
                            best_edge[i+1 + ' ' + nex] = (i + ' ' + prev)
