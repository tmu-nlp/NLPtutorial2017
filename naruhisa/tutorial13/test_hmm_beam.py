from collections import defaultdict
import numpy as np

P_T = dict()
P_E = dict()
p_tags = set()
V = 1000000
la = 0.95
la_unk = 1 - la
B = 32
with open('../tutorial04/result.txt', 'r') as f:
    for line in f:
        line = line.split()

        if(line[0] == 'T'):
            P_T[line[1] + ' ' + line[2]] = float(line[3])
            p_tags.add(line[1])
            p_tags.add(line[2])
        elif(line[0] == 'E'):
            P_E[line[1] + ' ' + line[2]] = float(line[3])

    p_tags = list(p_tags)
    print(p_tags)
f = open('my_answer.pos', 'w')
with open('../../data/wiki-en-test.norm', 'r') as f2:
    for line in f2:
        words = line.split()
        words.append('</s>')
        l = len(words)
        active_tags = defaultdict(list)
        best_score = defaultdict(lambda: 100000000)
        best_edge = defaultdict(lambda: None)
        best_score['0 <s>'] = 0
        best_edge['0 <s>'] = None
        active_tags[0] = ['<s>']
        for i in range(l):
            mybest = dict()
            for Prev in active_tags[i]:
                for Next in p_tags:
                    # print(Prev, Next)
                     if(str(i) + ' ' + Prev in best_score):
                        P = float(la_unk / V)
                        if(words[i] + ' ' + Next in P_E):
                            P += la * P_E[words[i] + ' ' + Next]
                        if Prev + ' ' + Next not in P_T:
                            #print('invalid')
                            continue
                            #P_T[Prev + ' ' + Next] = la_unk / V
                        score = float(best_score[str(i) +  ' ' + Prev] + -np.log(P_T[Prev + ' ' + Next]) + -np.log(P))
                        j = i + 1
                        if best_score[str(j) + ' ' + Next] > score or str(j) + ' ' + Next not in best_score:
                            best_score[str(j) + ' ' + Next] = score
                            best_edge[str(j) + ' ' + Next] = str(i) + ' ' + Prev
                            mybest[Next] = score
            for k, v in sorted(mybest.items(), key = lambda x:x[1]):
                print(k, v)
                active_tags[i+1].append(k)
                if len(active_tags[i+1]) == B:
                    break
            #print(active_tags[i+1])


        tags = []
        next_edge = best_edge[str(j) + ' </s>']
        while next_edge != '0 <s>':
            position, tag = next_edge.split()
            tags.append(tag)
            next_edge = best_edge[next_edge]
        tags.reverse()
        tagtag = ' '.join(tags)
        f.write(tagtag + '\n')
