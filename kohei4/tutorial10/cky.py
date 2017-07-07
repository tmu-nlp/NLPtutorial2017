import random
from collections import Counter, defaultdict
import math
from nltk.tree import Tree
import sys

def read_grammer(g_file):
    nonterm = []
    preterm  = defaultdict(lambda:[])
    with open(g_file, 'rt') as ff:
        for line in ff:
            #print(line.rstrip().split('\t'))
            rule = line.rstrip().split('\t')
            lhs, rhs, prob = rule[0], rule[1], -math.log(float(rule[2]))
            #print(lhs, rhs, prob)
            rhs_sym = rhs.split()
            #print(rbs_sym)
            if len(rhs_sym) == 1:
                preterm[rhs].append((lhs,prob))
            else:
                nonterm.append([lhs,rhs_sym[0],rhs_sym[1],prob])

    return nonterm, preterm


def print_tree(sym_ij, words, best_edge):
    (sym, i, j) = sym_ij
    #print(sym, i, j)
    if sym_ij in best_edge:
        edge = best_edge[sym_ij]
        return "("+sym+" "+print_tree(edge[0],words,best_edge) + " " + print_tree(edge[1],words,best_edge) + ")"
    else:
        return "(" +sym+" "+ words[i] + ")"


if __name__ == "__main__":

    #g_file = '../../test/08-grammar.txt'
    #input_file = '../../test/08-input.txt'
    g_file = '../../data/wiki-en-test.grammar'
    input_file = '../../data/wiki-en-short.tok'
    magic_number = 10**30
    nonterm, preterm = read_grammer(g_file)
    #print(nonterm,'\n', preterm)


    with open(input_file,'rt') as ff:
        for line in ff:
            words = line.strip().split(' ')
            #print(preterm[words[0]])
            best_score = defaultdict(lambda:magic_number)
            best_edge = dict()

            for i in range(len(words)):
                for (lhs, prob) in preterm[words[i]]:
                    #print(lhs,prob)
                    if prob < magic_number:
                        best_score[lhs,i,i+1] = prob
            #print(best_score.items())


            #print(words)
            for j in range(2,len(words)+1):
                for i in range(j-2,-1,-1):
                    for k in range(i+1,j):
                        #print(j,i,k)
                        for sym,lsym,rsym, logprob in nonterm:
                            #print(sym, lsym, rsym, logprob)
                            #print(best_score.items())

                            #if ((lsym,i,k) in best_score) and ((rsym,k,j) in best_score):
                            if best_score[lsym,i,k] < magic_number and best_score[rsym,k,j]< magic_number:
                                my_lp = best_score[lsym,i,k] + best_score[rsym,k,j] + logprob
                                #print(my_lp,best_score[sym,i,j])

                                if my_lp < best_score[sym,i,j]:
                                    #print('best edge')
                                    best_score[sym,i,j] = my_lp
                                    best_edge[sym,i,j] = (lsym,i,k),(rsym,k,j)

            #print(best_edge)

            S = print_tree(('S',0,len(words)),words,best_edge)
            print(S)

            t =Tree.fromstring(S)
            t.pretty_print()
