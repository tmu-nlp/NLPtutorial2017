from collections import defaultdict
import math
transition = defaultdict(int)
emission = defaultdict(int)
possible_tags = defaultdict(int)
V = 10 ** 6
lambda1=.95
lambda_unk=1-lambda1

beam = int(input('beam幅'))

with open('model_hmm.txt', 'r') as model:
    for line in model:
        line = line.split(' ')
        hmm_type = line[0]
        hmm_context = line[1]
        hmm_word = line[2]
        hmm_prob = float(line[3])
        possible_tags[hmm_context] = 1
        if hmm_type == 'T':
            transition[hmm_context +' ' + hmm_word] = hmm_prob
        else:
            emission[hmm_context +' '+ hmm_word] = hmm_prob
"""            
for key,value in transition.items():
    print('transition : {}\t{}'.format(key,value))
for key,value in emission.items():
    print('emission : {}\t{}'.format(key,value))
"""
#counter = 0
with open('../../data/wiki-en-test.norm', 'r') as test, open('my_answer.pos', 'w') as ans_file:
    for line in test:
        #counter = 0
        #前向きステップ
        words = line.strip().split()
        l = len(words)
        best_score = dict()
        best_edge = dict()
        active_tags = defaultdict(list)
        active_tags[0] = ['<s>']
        best_score['0 <s>'] = 0
        best_edge['0 <s>'] = None
        for i in range(l):
            my_best = dict()
            for prev in active_tags[i]:
                for next_tag in possible_tags.keys():
                    if (str(i)+' '+prev) in best_score and (prev+' '+next_tag) in transition:
                    #if best_score[str(i)+' '+prev] and transition[prev +' '+ next_tag]:
                        score = best_score[str(i)+' '+prev] + -math.log2(transition[prev+' '+next_tag])
                        P_emission = lambda_unk/V
                        if (next_tag+' '+words[i]) in emission:
                        #if emission[next_tag+' '+words[i]]:
                            P_emission += lambda1 * emission[next_tag+' '+words[i]]
                        score += -math.log2(P_emission) 
                        #if (str(i+1)+ ' '+ next_tag) not in best_score or best_score[str(i+1)+ ' '+ next_tag] < score:   
                        #if not best_score[str(i+1)+ ' '+ next_tag] or best_score[str(i+1)+ ' '+ next_tag] < score: 
                        if (str(i+1)+' '+ next_tag) not in best_score or best_score[str(i+1)+' '+ next_tag] > score:   
                            best_score[str(i+1)+' '+ next_tag] = score
                            best_edge[str(i+1)+' '+next_tag] = '{} {}'.format(i, prev)
                            my_best[next_tag] = score 
            for count,item in enumerate(sorted(my_best.items(), key=lambda x:x[1])):
                if count == beam:
                    break
                active_tags[i+1].append(item[0])
        #文末処理                    
        for prev in active_tags[l]:
            #print ((str(l-1)+' '+prev) in best_score)
            #print ((prev+' </s>') in transition)
            if (str(l)+' '+prev) in best_score and (prev+' </s>') in transition:
                #print ('koko')
            #if best_score[str(i)+' '+prev] and transition[prev+' </s>']:
                score = best_score[str(l)+' '+prev]+ -math.log2(transition[prev+' </s>'])
                if (str(l+1)+' </s>') not in best_score or best_score[str(l+1)+' </s>'] > score:
                #if not best_score[str(l)+' </s>'] or best_score[str(l)+' </s>'] < score:
                    best_score[str(l+1)+' </s>'] = score
                    best_edge[str(l+1)+' </s>'] = '{} {}'.format(l, prev)
                    #print ('count'+str(counter))
                    #print (str(l)+' </s>')
        #print ('直前'+str(l))            
        #後ろ向きステップ
        tags = list()
        next_edge = best_edge[str(l+1)+' </s>']
        while next_edge != '0 <s>':
        #while next_edge is not None:
            position = next_edge.split()[0]
            tag = next_edge.split()[1]
            tags.append(tag)
            next_edge = best_edge[next_edge]
        tags.reverse()
        #print (' '.join(tags).strip(' ')) 
        ans_file.write(' '.join(tags).strip(' '))                
        ans_file.write('\n')
        #exit()
