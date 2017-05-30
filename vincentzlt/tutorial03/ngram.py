
# coding: utf-8

# In[1]:

import sys,math,pprint
from collections import defaultdict


# In[63]:

class ngram:
    def __init__(self, f_name, n=1):
        self.n = n
        self.frq_dict = {}
        self.vocab_size_dict = defaultdict(lambda: 0)

        for i in range(self.n):
            self.frq_dict[str(i + 1)] = defaultdict(lambda: 0)

        for line in open(f_name, "r", encoding="UTF-8"):
            line = line.split()
            line.append("</s>")
            line.insert(0, "<s>")
            for i in range(self.n):
                _line_tp = self.line_tp(line, i + 1)
                for j in _line_tp:
                    self.frq_dict[str(i + 1)][" ".join(j)] += 1

        for i in range(self.n):
            for w in self.frq_dict[str(i + 1)]:
                self.vocab_size_dict[str(i + 1)] += self.frq_dict[str(i + 1)][
                    w]

    def line_tp(self, line, n):
        _tp_list = []
        len_line = len(line)
        for i in range(n):
            _tp_list.append(line[i:len_line - n + i + 1])
        return list(zip(*_tp_list))

    def calc_ngram_P(self, ngram_str):
        n=len(ngram_str.split())
        ngram_str_m1=" ".join(ngram_str.split()[:-1])
        if n == 1:
            lambda_ = 0.95
            vocab_size = 1000000
            P = self.frq_dict[str(n)][ngram_str] / float(
                self.vocab_size_dict[str(n)])
            return lambda_ * P + (1 - lambda_) / vocab_size
        else:
            lambda_n_m1=self.calc_lambda_n_m1(ngram_str)
            try:
                P=self.frq_dict[str(n)][ngram_str]/float(self.frq_dict[str(n)][ngram_str_m1])
            except ZeroDivisionError:
                P=0
            return lambda_n_m1*P+(1-lambda_n_m1)*self.calc_ngram_P(ngram_str.split()[-1])
            
    def calc_lambda_n_m1(self,ngram_str):
        
        n=len(ngram_str.split())
        try:
            c_w_m1_freq=self.frq_dict[str(n)][ngram_str]
        except KeyError:
            print(ngram_str)
            input()
        ngram_str_m1=" ".join(ngram_str.split()[:-1])
        
        u_w_m1_count= len(set([u for u in self.frq_dict[str(n)] if u.split()[:-1]==ngram_str.split()[:-1]]))
        return 1-u_w_m1_count/float(u_w_m1_count+c_w_m1_freq)
    
    def word_seg_viterbi(self,line,n_gram_model=1):
        line=line.replace("\n","")
        len_line=len(line)
        best_edge=[None]*(len_line+1)
        best_score=[0]*(len_line+1)
        ng_list=list(self.frq_dict[str(n_gram_model)].keys())
        
        for w_end in range(1,len_line+1):
            best_score[w_end]=10000000000
            for w_begin in range(0,w_end):
                word=line[w_begin:w_end]
                if (word in ng_list or len(word)==1) and word:
                    P_log_ng=-math.log(self.calc_ngram_P(word))
                    score_=best_score[w_begin]+P_log_ng
                    if score_<best_score[w_end]:
                        best_score[w_end]=score_
                        best_edge[w_end]=(w_begin,w_end)
        words=[]
        next_edge=best_edge[len(best_edge)-1]
        while next_edge:
            word=line[next_edge[0]:next_edge[1]]
            words.append(word)
            next_edge=best_edge[next_edge[0]]
        words.reverse()
        return words
    
    def word_seg_viterbi_file(self,f_name,n_gram_model=1):
        for line in open(f_name,"r",encoding="UTF-8"):
            _=self.word_seg_viterbi(line.replace("　",""))
            print(" ".join(_))



# In[65]:
if __name__=="__main__":
    _=ngram("./wiki-ja-train.word",1)
    _.word_seg_viterbi_file("./wiki-ja-test.txt")

