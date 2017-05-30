
# coding: utf-8

# In[1]:

import sys
import pprint
import math
from collections import defaultdict


# In[94]:

class w_pos():

    def __init__(self, f_name):
        self.pos_freq = {1: defaultdict(lambda: 0), 2: defaultdict(lambda: 0)}
        self.w_pos_freq = defaultdict(lambda: 0)

        for line in open(f_name, "r", encoding="UTF-8"):
            line = "<s>_<s> " + line.replace("\n", "") + " </s>_</s>"
            pos_line = []
            for w_pos in line.split(" "):
                w_, pos_ = w_pos.split("_")
                pos_line.append(pos_)
                self.w_pos_freq[w_pos] += 1
            for pos_ in pos_line:
                self.pos_freq[1][pos_] += 1
            for pos_ in zip(pos_line, pos_line[1:]):
                self.pos_freq[2][" ".join(pos_)] += 1

        self.pos_transission_matrice = dict()

        for i in self.pos_freq[1]:
            self.pos_transission_matrice[i] = dict()
            for j in self.pos_freq[1]:
                self.pos_transission_matrice[i][
                    j] = self.calc_HMM_transition_P(j, i)

    def calc_HMM_transition_P(self, y_i, y_prev):
        return self.pos_freq[2][y_prev + " " + y_i] / float(self.pos_freq[1][
            y_prev])

    def calc_HMM_emission_P(self, w, tag):
        lambda_ = 0.91
        vocab_size = 10000000
        P = self.w_pos_freq[w + "_" + tag] / float(self.pos_freq[1][tag])
        return lambda_ * P + (1 - lambda_) / float(vocab_size)

    def calc_POS_viterbi(self, line_str):
        line = line_str.replace("\n","").split(" ")
        line.append("</s>")
        line.insert(0, "<s>")
        len_line = len(line)
        best_score = dict()
        best_node = dict()
        n_ = 0
        for w_ in line:
            w = str(n_) + "_" + w_

            best_score[w] = dict()
            best_node[w] = dict()
            if w_ == "<s>":
                best_score[w][str(n_) + "_<s>"] = 0
                best_node[w][str(n_) + "_<s>"] = None
            elif w_ == "</s>":
                best_score[w][str(n_) + "_</s>"] = 1000000.0
                best_node[w][str(n_) + "_</s>"] = 0
            else:
                for tag in self.pos_freq[1]:
                    if tag != "<s>" and tag != "</s>":
                        best_score[w][str(n_) + "_" + tag] = 1000000.0
                        best_node[w][str(n_) + "_" + tag] = 0
            n_ += 1
        # pprint.pprint(best_score)
        # pprint.pprint(best_node)
        n_ = 1
        for w, w_prev in zip(line[1:], line):
            for tag_ in best_score[str(n_) + "_" + w]:
                tag = tag_.split("_")[1]
                for tag_prev_ in best_score[str(n_ - 1) + "_" + w_prev]:
                    tag_prev = tag_prev_.split("_")[1]

                    pos_trans_P = self.calc_HMM_transition_P(tag, tag_prev)
                    pos_emiss_P = self.calc_HMM_emission_P(w, tag)
                    
                    if pos_trans_P == 0 and pos_emiss_P == 0:
                        _ = 0
                    elif pos_trans_P == 0:
                        _ = -math.log(pos_emiss_P)
                    elif pos_emiss_P == 0:
                        _ = -math.log(pos_trans_P)
                    else:
                        _ = -math.log(pos_trans_P) - math.log(pos_emiss_P)
                    prev_score = best_score[str(n_ - 1) + "_" + w_prev][str(n_ - 1) + "_" +tag_prev]
                    _+=prev_score
                    if _ < best_score[str(n_) + "_" + w][str(n_) + "_" + tag]:
                        best_score[str(n_) + "_" + w][str(n_) + "_" + tag] = _
                        best_node[str(n_) + "_" + w][str(n_) + "_" + tag] = str(n_ - 1) + "_" + tag_prev
            n_ += 1
        # pprint.pprint(best_score)
        # pprint.pprint(best_node)
        n_ -= 1
        tags = []
        next_node = best_node[str(n_) + "_</s>"][str(n_) + "_</s>"]
        while next_node != "0_<s>":
            n_ -= 1
            ind, tag = next_node.split("_")
            tags.append(tag)

            next_node = best_node[str(n_) + "_" + line[int(ind)]][next_node]
        tags.reverse()
        return " ".join(tags)

    def calc_POS_viterbi_f(self, f_name):
        with open("./my-answer.pos", "w", encoding="UTF-8") as f:
            pass
        for line in open(f_name,"r",encoding="UTF-8"):
            _=self.calc_POS_viterbi(line)+"\n"
            with open("./my-answer.pos","a",encoding="UTF-8") as f:
                f.write(_)
        # In[95]:

temp = w_pos(f_name="./wiki-en-train.norm_pos")


# In[97]:

temp.calc_POS_viterbi_f("..//..//data//wiki-en-test.norm")


# In[ ]:
