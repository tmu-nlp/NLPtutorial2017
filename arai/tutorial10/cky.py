import math
from collections import defaultdict

def subroutine_print(sym_ij):
    if sym_ij in best_edge:
        #print(best_edge)
        sym, i, j = sym_ij.split()
        return('({} {} {})'.format(sym, subroutine_print(best_edge[sym_ij][0]),subroutine_print(best_edge[sym_ij][1])))
    else:
        #print(best_edge)
        sym, i, j = sym_ij.split()
        #print(sym_ij)
        return('({} {})'.format(sym, words[int(i)]))

nonterm = []
preterm = defaultdict(list)

for rule in open('../../data/wiki-en-test.grammar'):
    lhs, rhs, prob = rule.split('\t')
    rhs_symbols = rhs.split()
    #print(rhs)
    
   # for rhs in rule:
    if len(rhs_symbols) == 1:
        preterm[''.join(rhs_symbols)].append((lhs, math.log(float(prob))))
    else:
        nonterm.append([lhs, rhs_symbols[0], rhs_symbols[1], math.log(float(prob))])
    #print(preterm)
for line in open('../../data/wiki-en-short.tok'):
    words = line.strip().split()
    best_score = defaultdict(lambda : float('-inf'))
    best_edge = defaultdict(int)
    for i in range(len(words)):
        #print(words)
        for lhs, log_prob in preterm[words[i]]:
            #print(preterm)
            #print(lhs)
            best_score['{} {} {}'.format(lhs, i, i + 1)] = log_prob

    for j in range(2, len(words) + 1):
        #print(j)
        for i in range(j-2, -1, -1):
            for k in range(i+1, j):
                #print(nonterm)
                for sym, lsym, rsym, logprob in nonterm:
                    #print(best_score)
                    if '{} {} {}'.format(lsym, i, k) in best_score and '{} {} {}'.format(rsym, k, j) in best_score:
                        #print(best_score)
                        my_lp = best_score['{} {} {}'.format(lsym, i, k)] + best_score['{} {} {}'.format(rsym, k, j)] + logprob
                        if my_lp > best_score['{} {} {}'.format(lsym, i, j)]:
                            best_score['{} {} {}'.format(sym, i, j)] = my_lp
                            best_edge['{} {} {}'.format(sym, i, j)] = ('{} {} {}'.format(lsym, i, k), '{} {} {}'.format(rsym, k, j))
    print(subroutine_print('S 0 {}'.format(len(words))))


