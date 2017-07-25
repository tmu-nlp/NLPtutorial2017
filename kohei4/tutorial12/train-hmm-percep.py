
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


def make_hmm_model(train_f,model_f):
    emit = defaultdict(lambda: 0)
    transition = defaultdict(lambda: 0)
    context = defaultdict(lambda: 0)

    with open(train_f,'r') as f:
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

    with open(model_f,'w') as ff:
        for key, value in transition.items():
            previous, word = key.split(" ")
            print("T {} {}".format(key, value/context[previous]),file=ff)
        for key, value in emit.items():
            previous, word = key.split(" ")
            print("E {} {}".format(key, value/context[previous]),file=ff)

    transition = defaultdict(int)
    emission = defaultdict(int)
    possible_tags = defaultdict(int)


    with open(model_f,'r') as f:
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

    return transition, emission, possible_tags

if __name__ == "__main__":

    input_file = '../../data/wiki-en-train.norm_pos'
    #input_file = '../../test/05-train-input.txt'
    model_f = 'HMM_model.txt'
    epock = 3

    w = defaultdict(float)

    transition, emission, possible_tags = make_hmm_model(input_file,model_f)


    with open(input_file,'r') as f:
        for _ in range(epock):
            list_l = list(f)
            random.shuffle(list_l)
            for line in list_l:
                X = []
                Y_pr = []
                wordtags =line.rstrip("\n").split(" ")
                for wordtag in wordtags:
                    x, y = wordtag.split("_")
                    X.append(x)
                    Y_pr.append(y)
                #print(X, Y_pr)
                Y_hat = hmm_viterbi(w,X)
                #print(Y_hat)
                phi_pr = create_features(X, Y_pr)
                #print(phi_prime)
                phi_hat = create_features(X, Y_hat)

                phi_memo = dict()

                for key,value in phi_pr.items():
                    if key in phi_hat:
                        phi_memo[key] = value - phi_hat[key]
                    else:
                        phi_memo[key] = value
                for key, value in phi_hat.items():
                    if key in phi_pr:
                        phi_memo[key] = phi_pr[key] - value
                    else:
                        phi_memo[key] = - value

                for key, value in phi_memo.items():
                    w[key] += value
    #print(w)
    #test_file = "../../test/05-test-input.txt"
    with open('net','wb') as ff:
        pickle.dump( (w, possible_tags, transition),ff)

"""
    test_file = '../../data/wiki-en-test.norm'
    with open(test_file,'r') as gg:
        for line in gg:
            words = list(line.strip().split())
            print(' '.join(hmm_viterbi(w,words)))
"""
