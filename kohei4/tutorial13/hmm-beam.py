#Kohei-no-MacBook-Air:tutorial04 kohei
#$ ../../script/gradepos.pl ../../data/wiki-en-test.pos myanswer.pos

#HMM tutorial 04
#Accuracy: 90.82% (4144/4563)

#B=3
#Accuracy: 90.51% (4130/4563)

#B=6
#Accuracy: 90.86% (4146/4563)


import sys
import math
from collections import defaultdict


B = 6

emit = defaultdict(lambda: 0)
transition = defaultdict(lambda: 0)
context = defaultdict(lambda: 0)


with open('../../data/wiki-en-train.norm_pos','r') as f:
#with open(sys.argv[1], 'r') as f:
    for line in f:
        line=line.rstrip("\n")
        previous = "<s>"
        context[previous] += 1
        wordtags = line.split(" ")
        for wordtag in wordtags:
            word, tag = wordtag.split("_")
            #print(word, tag)
            transition[previous+ " " + tag] += 1
            context[tag] += 1
            emit[tag+" " + word] += 1
            previous = tag
        transition[previous+" </s>"] += 1

#print(transition.items())

with open('HMM_model.txt','w') as ff:
    for key, value in transition.items():
        previous, word = key.split(" ")
        print("T {} {}".format(key, value/context[previous]),file=ff)
    for key, value in emit.items():
        previous, word = key.split(" ")
        print("E {} {}".format(key, value/context[previous]),file=ff)

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
        active_tags = dict()
        best_score["0 <s>"] = 0
        best_edge["0 <s>"] = "NULL"
        active_tags = [["<s>"]]

        for i in range(l):
            my_best = dict()
            for prev in active_tags[i]:
                #print(prev)
                for NEXT in possible_tags.keys():
                    #print(NEXT)
                    #if Prob_E(words[i]+" "+NEXT) and ((str(i)+" "+ prev in best_score) and transition[prev+" "+NEXT]):
                    if ((str(i)+" "+ prev in best_score) and transition[prev+" "+NEXT]):
                        score = best_score[str(i)+" "+ prev] + -math.log(transition[prev + " " + NEXT]) \
                        + -math.log(Prob_E(NEXT+" "+words[i]))
                        #print(prev,NEXT,score)
                        #print(((str(i+1)+" "+NEXT) in best_score))
                        if ((str(i+1)+" "+NEXT) in best_score) and best_score[str(i+1)+" "+NEXT] > score:
                            #print("update")
                            best_score[str(i+1)+" "+NEXT] = score
                            best_edge[str(i+1)+" "+NEXT] = str(i)+" "+prev
                            my_best[NEXT] = score

                        elif (str(i+1)+" "+NEXT in best_score) == False :
                            #print("new")
                            best_score[str(i+1)+" "+NEXT] = score
                            best_edge[str(i+1)+" "+NEXT] = str(i)+" "+prev
                            my_best[NEXT] = score

            best_B = sorted(my_best.items(), key=lambda x:x[1])[:B]
            best_tag = []
            for x, y in best_B:
                best_tag.append(x)
            #print(best_tag)
            active_tags.append(best_tag)


        #for prev in possible_tags.keys():
        for prev in active_tags[l]:
            if transition[prev+" </s>"]:
                #print(prev)
                score = best_score[str(l)+" "+prev] + -math.log(transition[prev + " </s>"])
                #print(score)

                if (str(l+1)+" </s>" in best_score) == False or best_score[str(l+1)+" </s>"] > score:
                    best_score[str(l+1)+" </s>"] = score
                    #print("</s>",score)
                    best_edge[str(l+1)+" </s>"] = str(l)+" "+prev




        tags =[]
        NEXT_edge = best_edge[str(l+1)+" </s>"]
        while NEXT_edge != "0 <s>":
            position, tag = NEXT_edge.split(" ")
            tags.append(tag)
            NEXT_edge = best_edge[NEXT_edge]
            #print(NEXT_edge)

        tags.reverse()
        print(" ".join(tags))
