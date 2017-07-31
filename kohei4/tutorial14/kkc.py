
"""
i: 273 (7.6193134245046%)
d: 37 (1.03265420039073%)
s: 771 (21.5182807703042%)
e: 2502 (69.8297516048004%)
WER: 32.66%
Prec: 70.56%
Rec: 75.59%
F-meas: 72.99%
Sent: 8.33%
"""

import sys
import math
from collections import defaultdict
import random
import pickle

def lm_load(lm_f):

    w_prob = defaultdict(float)
    t_cnt = 0

    with open(lm_f,'r') as f:
    #with open(sys.argv[1], 'r') as f:
        for line in f:
    #        line = line.lower()

            w_list = line.split('\t')
            w_prob[w_list[0]] = float(w_list[1])

    #print(w_prob.items())
    return w_prob


def Plm(c_p_word):

    r1 = 0.95
    r_unk = 1-r1
    V = 1000000
    P = r_unk / V
    if P_lm_ba[c_p_word] != 0.0:
        P += r1 * P_lm_ba[c_p_word]
    #print(c_p_word,P)
    return P

def get_tag(index, score):
    tmp_score, word = None, ""
    for w, s in score[index].items():
        if (tmp_score is None) or (s < tmp_score):
            tmp_score = s
            tag = w
    return tag

def kkc_viterbi(tar_file,):


    with open(tar_file,'r') as f:
        for line in f:
            cols=line.strip()
            edge = defaultdict(lambda:dict())
            score = defaultdict(lambda:dict())
            edge[0]["<s>"] = "NULL"
            score[0]["<s>"] = 0

            #print(cols)
            for end in range(1,len(cols)+1):
                score[end]
                edge[end]
                for begin in range(end):
                    pron=cols[begin:end]
                    #print(pron)
                    my_tm = tm[pron]
                    if not my_tm and len(pron) == 1:
                        my_tm[pron] = 10**-60
                        #my_tm[pron] = 1
                    #print(my_tm)
                    for curr_word, tm_prob in my_tm.items():
                        #print(curr_word,tm_prob)
                        for prev_word, prev_score in score[begin].items():
                            #print(prev_word, prev_score)
                            #print(tm_prob, Plm(prev_word+" "+curr_word))
                            curr_score = prev_score \
                            -math.log(tm_prob * Plm(prev_word+" "+curr_word))

                            if (score[end].get(curr_word) is None) \
                            or (curr_score < score[end][curr_word]):

                                score[end][curr_word] = curr_score
                                edge[end][curr_word]=(begin,prev_word)
                                #edge[end][curr_word]=(begin,end)
                                #print("here", score[end][curr_word])

            #print(edge)
            #print(score)
            tags =[]

            tag = get_tag(len(cols),score)
            NEXT_edge = edge[len(cols)][tag]
            #print(tag)
            #print(NEXT_edge)
            while NEXT_edge != 'NULL':
                tags.append(tag)
                tag = get_tag(NEXT_edge[0],score)
                #print(tag)
                NEXT_edge = edge[NEXT_edge[0]][tag]
                #print(NEXT_edge)
            tags.reverse()
            print(" ".join(tags))


if __name__ == '__main__':

    tm = defaultdict(lambda:dict())

    #lm_file = 'lm_t.txt'
    #tm_file = 'tm_t.txt'
    #tar_file = '../../test/06-pron.txt'
    lm_file = 'lm.txt'
    tm_file = 'tm.txt'
    tar_file = '../../data/wiki-ja-test.pron'

    r = 0.95
    V = 1000000


    with open(tm_file,'r') as ff:
        for line in ff:
            cols = line.strip().split(" ")
            if cols[0] == "E":
                tm[cols[2]][cols[1]]=float(cols[3])

    #print(tm)

    P_lm_ba = lm_load(lm_file)
    #print(P_lm_ba['行 の'])
    #print(P_lm_ba['行 が'])

    kkc_viterbi(tar_file)
