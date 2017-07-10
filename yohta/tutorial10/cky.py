import math
from collections import defaultdict


def print_func(sym_ij):
    sym_,i_,j_ = sym_ij.split(' ')
    if sym_ij in best_edge:
        return '(' + sym_ + ' ' + print_func(best_edge[sym_ij][0]) + ' ' + print_func(best_edge[sym_ij][1]) + ')'
    else:
        return '(' + sym_ + ' ' + words[int(i_)] + ')'


nonterm = []
preterm = defaultdict(list)
#i_f = open('../../test/08-input.txt')
i_f = open('../../data/wiki-en-short.tok')
#g_f = open('../../test/08-grammar.txt')
g_f = open('../../data/wiki-en-test.grammar')
def read_grammar(g_f):
    for rule in g_f:
        lhs,rhs,prob = rule.split('\t')
        rsym = rhs.split(' ')
        if len(rsym) == 1:
            preterm[rhs].append((lhs,math.log(float(prob)))) # タプルのリスト
        else:
            nonterm.append([lhs,rsym[0],rsym[1],math.log(float(prob))]) # リストのリスト
    return preterm,nonterm
#print('{}\n{}'.format(preterm,nonterm))

mul = -(10**10)

if __name__ == '__main__':
    for line in i_f:
        best_score = defaultdict(lambda :mul)
        best_edge = dict()
        words = line.strip().split(' ')
        preterm,nonterm = read_grammar(g_f)
        for i in range(len(words)):
            for lhs,log_prob in preterm[words[i]]:
                best_score['{} {} {}'.format(lhs,i,i+1)] = log_prob
#        print(best_score)
        for j in range(2,len(words)+1): # 2 ~ wordsの数-1
            for i in reversed(range(0,j-1)):
                for k in range(i+1,j):
                    for sym,lsym,rsym,logprob in nonterm:
#                        print(lsym)
#                        print('{}\t{}\t{}'.format(j,i,k))
                        if best_score['{} {} {}'.format(lsym,i,k)] > mul and best_score['{} {} {}'.format(rsym,k,j)] > mul:
                            my_lp = best_score['{} {} {}'.format(lsym,i,k)] + best_score['{} {} {}'.format(rsym,k,j)] + logprob
                            if my_lp > best_score['{} {} {}'.format(sym,i,j)]:
                                best_score['{} {} {}'.format(sym,i,j)] = my_lp
                                best_edge['{} {} {}'.format(sym,i,j)] = ('{} {} {}'.format(lsym,i,k),'{} {} {}'.format(rsym,k,j))
        print(print_func('S 0 {}'.format(len(words))))
        #　未知語の処理が謎
