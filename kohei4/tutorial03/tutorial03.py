import sys
import math
from collections import defaultdict

w_prob = defaultdict(lambda: 0)


with open('model_file.txt','r') as f:
#with open(sys.argv[1], 'r') as f:
    for line in f:
#        line = line.lower()
        w_list = line.split('\t')
        w_prob[w_list[0]] = float(w_list[1])

#def w_prob(word):
#    r1 = 0.95
#    r_unk = 1-r1
#    V = 1000000
#    P = r_unk / V
#    if w_prob[word]:
#        P += r1 * w_prob[w]
#
#    return P


#print(w_prob)
best_edge=[]
with open(sys.argv[1], 'r') as f2:
    for line in f2:
        #print(len(line))
        best_score=[10**10]*len(line)
        best_edge=[10**10]*len(line)
        #print(best_scope)
        best_edge[0] = 'NULL'
        best_score[0] = 0
        for i in range(1,len(line)):
            best_score[i] = 10**10
            for j in range(len(line)-1):
                word = line[j:i]
                if (word in w_prob) or len(word) == 1:
                    r_unk = 1 - 0.95
                    prob = r_unk / (10**5)
                    if w_prob[word]:
                        prob += 0.95 * w_prob[word]

                    #print(word)
                    #print(prob)
                    my_score = best_score[j] - math.log(prob)
                    if my_score < best_score[i]:
                        best_score[i] = my_score
                        best_edge[i] = (j, i)

        words = []
        next_edge = best_edge[len(best_edge) - 1]
        while  next_edge != 'NULL':
            word = line[next_edge[0]:next_edge[1]]
            words.append(word)
            next_edge = best_edge[next_edge[0]]
        words.reverse()

        print(" ".join(words))
