#python cky.py ../../data/wiki-en-test.grammar ../../data/wiki-en-short.tok wiki-en-test.trees
#python ../../script/print-trees.py < wiki-en-test.trees

from collections import defaultdict
import sys
import math

def penn_treebank(symij, words, best_edge):
    sym, i, j = symij.split(" ")
    if symij in best_edge:
        return "({} {} {})".format(sym, penn_treebank(best_edge[symij][0], words, best_edge), penn_treebank(best_edge[symij][1], words, best_edge))
    elif sym != "S":
        return "({} {})".format(sym, words[int(i)])
    else:
        return "(S {})".format("_".join(words))

def cky(grammar_path, fin_path, fout_path):
    nonterm = list()
    preterm = defaultdict(list)

    with open(grammar_path, "r") as fg:
        for rule in fg:
            lhs, rhs, prob = rule.strip("\n").split("\t")
            rhs_symbols = rhs.split(" ")
            if len(rhs_symbols) == 1:
                preterm[rhs].append((lhs, math.log(float(prob), 2)))
            else:
                nonterm.append((lhs, rhs_symbols[0], rhs_symbols[1], math.log(float(prob), 2)))

    with open(fin_path, "r") as fin, open(fout_path, "w") as fout:
        for line in fin:
            words = line.strip("\n").split(" ")
            best_score = defaultdict(lambda: 10 ** 10)
            best_edge = dict()
            
            for i in range(0, len(words)):
                for lhs, logprob in preterm[words[i]]:
                    best_score["{} {} {}".format(lhs, i, i + 1)] = logprob

            for j in range(2, len(words) + 1):
                for i in range(j - 2, -1, -1):
                    for k in range(i + 1, j):
                        for sym, lsym, rsym, logprob in nonterm:
                            if "{} {} {}".format(lsym, i, k) in best_score and "{} {} {}".format(rsym, k, j) in best_score:
                                my_lp = best_score["{} {} {}".format(lsym, i, k)] + best_score["{} {} {}".format(rsym, k, j)] - logprob
                                if my_lp < best_score["{} {} {}".format(sym, i, j)]:
                                    best_score["{} {} {}".format(sym, i, j)] = my_lp
                                    best_edge["{} {} {}".format(sym, i, j)] = ("{} {} {}".format(lsym, i, k), "{} {} {}".format(rsym, k, j))

            fout.write("{}\n".format(penn_treebank("S 0 {}".format(len(words)), words, best_edge)))

if __name__ == "__main__":
    cky(sys.argv[1], sys.argv[2], sys.argv[3])
