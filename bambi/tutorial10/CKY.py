import math
from collections import defaultdict
nonterm = []
preterm = defaultdict(list)
#grammar_file = "../test/08-grammar.txt"
grammar_file = "../../data/wiki-en-test.grammar"
for rule in open(grammar_file):
    lhs,rhs,prob = rule.split("\t")
    prob = float(prob)
    rhs_symbols = rhs.split()
    if len(rhs_symbols) == 1:
        # len(rhs_symbols) -> one word so use "rhs", ok ...split() no effect
        preterm[rhs].append((lhs, math.log(prob,2))) # key = word, value = pos,prob
    else:
        nonterm.append((lhs,rhs_symbols[0],rhs_symbols[1],math.log(prob,2)))
#token_file = "../test/08-input.txt"
token_file = "../../data/wiki-en-short.tok"
def subroutine(sym_ij):
    #print(sym_ij)
    sym,i,j = sym_ij.split("\t")
    if sym_ij in best_edge:
        return "({} {} {})".format(sym,subroutine(best_edge[sym_ij][0]),subroutine(best_edge[sym_ij][1]))
    else: # for terminals
        return "({} {})".format(sym,words[int(i)])

def key(*names):
    return "\t".join(map(str,names))

for line in open(token_file):
    best_score = defaultdict(lambda: float('-inf')) # index: symbol(i,j) value = best log prob
    best_edge = dict() #  index: symbol(i,j) value = (lsym(i,k), rsym(k,j))
    words = line.strip("\n").split()
    #  Add Pre-Terminals
    for i in range(len(words)):
        P = preterm[words[i]]
        for lhs, log_prob in P:
            best_score[key(lhs,i,i+1)] = log_prob
    # Combine Non-Terminals
    for j in range(2,len(words)+1):
        for i in range(j-2,-1,-1):
            for k in range(i+1, j):
                for sym, lsym, rsym, logprob in nonterm:
                    # Both children must have a probability
                    if best_score[key(lsym,i,k)] > -math.inf and best_score[key(rsym,k,j)] > -math.inf:
                        # Find the log probability for this node/edge
                        my_lp = best_score[key(lsym,i,k)] + best_score[key(rsym,k,j)] + logprob
                        # If this is the best edge, update
                        if my_lp > best_score[key(sym,i,j)]:
                            best_score[key(sym,i,j)] = my_lp
                            best_edge[key(sym,i,j)] = (key(lsym,i,k), key(rsym,k,j))
    # Print the “S” that spans all words
    print(subroutine(key("S",0,len(words))))
