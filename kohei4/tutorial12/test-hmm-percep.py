"""
kohei$ python train-hmm-percep.py
kohei$ python test-hmm-percep.py >answer4.txt
kohei$ ../../script/gradepos.pl ../../data/wiki-en-test.pos answer4.txt
Accuracy: 82.80% (3778/4563)
"""


import sys
import math
from collections import defaultdict
import random
import pickle

def hmm_viterbi(w,word):
    #print(X)
    l = len(word)
    best_score = dict()
    best_edge = dict()
    best_score["0 <s>"] = 0
    best_edge["0 <s>"] = "NULL"


    for i in range(l):
        phi = defaultdict(float)
        for prev in possible_tags.keys():
            for NEXT in possible_tags.keys():
                if ((str(i)+" "+ prev in best_score) and transition[prev+" "+NEXT]):
                    #score = best_score[str(i)+" "+ prev] + -math.log(transition[prev + " " + NEXT]) + -math.log(Prob_E(NEXT+" "+words[i]))
                    phi.update(create_t(prev,NEXT))
                    phi.update(create_e(NEXT, word[i]))

                    score_t = phi["T,"+ prev + "," + NEXT] * w["T,"+ prev + "," + NEXT]
                    score_e = phi["E,"+ NEXT + "," + word[i]] *w["E,"+ NEXT + "," + word[i]]

                    if (NEXT == "NNP") and (word[i][0].isupper()):
                        score_e += phi["CAPS,"+ NEXT] * w["CAPS," +NEXT]

                    score = best_score[str(i)+" "+ prev] \
                    + score_t + score_e

                    if ((str(i+1)+" "+NEXT) in best_score) and \
                    best_score[str(i+1)+" "+NEXT] < score:
                        #print("update")
                        best_score[str(i+1)+" "+NEXT] = score
                        best_edge[str(i+1)+" "+NEXT] = str(i)+" "+prev
                    elif (str(i+1)+" "+NEXT in best_score) == False :
                        #print("new")
                        best_score[str(i+1)+" "+NEXT] = score
                        best_edge[str(i+1)+" "+NEXT] = str(i)+" "+prev

    #print(best_score)

    for prev in possible_tags.keys():
        if transition[prev+" </s>"]:
            #score = best_score[str(l)+" "+prev] + -math.log(transition[prev + " </s>"])
            phi.update(create_t(prev,"</s>"))
            score_t = phi["T,"+ prev + "," +"</s>"] * w["T,"+ prev + "," +"</s>"]
            score = best_score[str(l)+" "+ prev] + score_t

            if (str(l+1)+" </s>" in best_score) == False or best_score[str(l+1)+" </s>"] < score:
                best_score[str(l+1)+" </s>"] = score
                #print("</s>",score)
                best_edge[str(l+1)+" </s>"] = str(l)+" "+prev

    #print(phi)

    tags =[]
    NEXT_edge = best_edge[str(l+1)+" </s>"]
    while NEXT_edge != "0 <s>":
        position, tag = NEXT_edge.split(" ")
        tags.append(tag)
        NEXT_edge = best_edge[NEXT_edge]
        #print(NEXT_edge)

    tags.reverse()
    #print(" ".join(tags))
    return tags

def create_t(X,Y):
    phi_t = defaultdict(float)
    phi_t["T,"+ X + "," + Y] = 1
    return phi_t

def create_e(X,Y):
    phi_e = defaultdict(float)
    phi_e["E,"+ X + "," + Y] = 1
    if (X == "NNP") and (Y[0].isupper()):
        #phi["CAPS,"+ X] += 1
        phi_e["CAPS,"+ X] = 1
    return phi_e

def create_features(X, Y):
    phi = defaultdict(float)
    for i in range(len(Y) + 1):
        if i == 0:
            first_tag = '<s>'
        else:
            first_tag = Y[i-1]

        if i == len(Y):
            next_tag = '</s>'
        else:
            next_tag = Y[i]
        phi['T,{},{}'.format(first_tag,next_tag)] += 1

    for i in range(len(Y)):
        phi['E,{},{}'.format(Y[i], X[i])] += 1
        if X[i][0].isupper() and Y[i] == "NNP":
            phi['CAPS,{}'.format(Y[i])] = +1
    return phi



if __name__ == "__main__":

    with open('net','rb') as ff:
        w, possible_tags, transition = pickle.load(ff)


    test_file = '../../data/wiki-en-test.norm'
    with open(test_file,'r') as gg:
        for line in gg:
            words = list(line.strip().split())
            print(' '.join(hmm_viterbi(w,words)))
