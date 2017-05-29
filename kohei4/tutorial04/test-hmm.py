#Kohei-no-MacBook-Air:tutorial04 kohei
#$ ../../script/gradepos.pl ../../data/wiki-en-test.pos myanswer.pos
#Accuracy: 90.82% (4144/4563)

#Most common mistakes:
#NNS --> NN	45
#NN --> JJ	27
#NNP --> NN	22
#JJ --> DT	22
#VBN --> NN	12
#JJ --> NN	12
#NN --> IN	11
#NN --> DT	10
#NNP --> JJ	8
#JJ --> RB	7


import sys
import math
from collections import defaultdict




transition = defaultdict(lambda: 0)
emission = defaultdict(lambda: 0)
possible_tags = defaultdict(lambda: 0)


with open('HMM_model.txt','r') as f:
#with open(sys.argv[1], 'r') as f:
    for line in f:
        line=line.rstrip("\n")
        #print(line)
        typ, context, word, prob = line.split(" ")
        prob=float(prob)
        possible_tags[context] =1
        if typ == "T":
            transition[context+ " " + word] = prob
        else:
            emission[context+ " " + word] = prob

#print(transition.items())
#print(emission.items())
#print(possible_tags.items())

def Prob_E(wordNEXT):
    r1 = 0.95
    r_unk = 1-r1
    V = 1000000
    P = r_unk / V
    if emission[wordNEXT]:
        P += r1 * emission[wordNEXT]

    return P


with open('../../data/wiki-en-test.norm','r') as f:
#with open(sys.argv[1], 'r') as f:
    for line in f:
        #print("\n"+line)
        line=line.rstrip("\n")
        words = line.split()
        l = len(words)
        best_score = dict()
        best_edge = dict()

        best_score["0 <s>"] = 0
        best_edge["0 <s>"] = "NULL"

        for i in range(l):
            for prev in possible_tags.keys():
                #print(prev)
                for NEXT in possible_tags.keys():
                    #print(NEXT)
                    #if Prob_E(words[i]+" "+NEXT) and ((str(i)+" "+ prev in best_score) and transition[prev+" "+NEXT]):
                    if ((str(i)+" "+ prev in best_score) and transition[prev+" "+NEXT]):
                        #print(transition[prev+" "+NEXT],Prob_E(words[i]+" "+NEXT))
                        #print(str(i)+" "+ prev)
                        #print(NEXT)
                        #print(NEXT+" "+words[i] , math.log(Prob_E(NEXT+" "+words[i])))
                        score = best_score[str(i)+" "+ prev] + -math.log(transition[prev + " " + NEXT]) + -math.log(Prob_E(NEXT+" "+words[i]))
                        #print(prev,NEXT,score)
                        #print(((str(i+1)+" "+NEXT) in best_score))
                        if ((str(i+1)+" "+NEXT) in best_score) and best_score[str(i+1)+" "+NEXT] > score:
                            #print("update")
                            best_score[str(i+1)+" "+NEXT] = score
                            best_edge[str(i+1)+" "+NEXT] = str(i)+" "+prev
                        elif (str(i+1)+" "+NEXT in best_score) == False :
                            #print("new")
                            best_score[str(i+1)+" "+NEXT] = score
                            best_edge[str(i+1)+" "+NEXT] = str(i)+" "+prev

        #print(best_score.items())
        #print(best_edge.items())


        #print("lastlast")
        for prev in possible_tags.keys():
            if transition[prev+" </s>"]:
                #print(prev)
                score = best_score[str(l)+" "+prev] + -math.log(transition[prev + " </s>"])
                #print(score)

                if (str(l+1)+" </s>" in best_score) == False or best_score[str(l+1)+" </s>"] > score:
                    best_score[str(l+1)+" </s>"] = score
                    #print("</s>",score)
                    best_edge[str(l+1)+" </s>"] = str(l)+" "+prev
                    #print(str(l)+" "+prev)

#        for prev in possible_tags.keys():
#            if (str(i)+" "+ prev in best_score) and transition[prev+" </s>"]:
#                score = best_score[str(l)+" "+prev] - math.log(transition[prev + " </s>"]) - math.log(Prob_E(" </s>"+" "+words[l-1]))
                #print((str(l+1)+" </s>" in best_score) == 0 )
                #print(score)
#                if (str(l+1)+" </s>" in best_score) == 0 or best_score[str(l+1)+" </s>"] < score:
#                    best_edge[str(l+1)+" </s>"] = str(l)+" "+prev


        #print(best_score.items())
        #print(best_edge.items())


        tags =[]
        NEXT_edge = best_edge[str(l+1)+" </s>"]
        while NEXT_edge != "0 <s>":
            position, tag = NEXT_edge.split(" ")
            tags.append(tag)
            NEXT_edge = best_edge[NEXT_edge]
            #print(NEXT_edge)

        tags.reverse()
        print(" ".join(tags))
