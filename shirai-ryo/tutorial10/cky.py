from collections import defaultdict
import math

def PRINT(symij):
    sym, ij = symij.replace(")", " ").split("(")
    i, j = ij.split(", ")
    if symij in best_edge: # 非終端記号
        return "({} {} {})".format(sym, PRINT(best_edge[symij][0]), PRINT(best_edge[symij][1]))
    else:
        # print(best_edge)
        # print(sym)
        # print(i)
        # print(j)
        return "({} {})".format(sym, words[int(i)])


nonterm = []
preterm = defaultdict(list)

# CKY疑似コード：文法の読み込み
with open('../../data/wiki-en-test.grammar') as grammar_file:
    for rule in grammar_file:
        rule = rule.split('\t')
        lhs = rule[0] # SとかVPとか
        rhs = rule[1] # NPとか、sawみたいな具体的な単語とか
        prob = float(rule[2]) # 確率
        # lhs　→　rhs の確率がprob
        rhs_symbol = rhs.split(" ")
        # print(rhs_symbol)

        if len(rhs_symbol) == 1: # 前終端記号
            preterm[rhs].append([lhs, math.log2(prob)])
        else: # 非終端記号
            nonterm.append([lhs, rhs_symbol[0], rhs_symbol[1], math.log(prob)])

# print(preterm)
# print("ここまで")
# print(nonterm)


# CKY疑似コード：前終端記号を追加
with open('../../data/wiki-en-short.tok') as input_file:
    for line in input_file:
        best_score = defaultdict(lambda: -10000000)
        best_edge = {}
        words = line.split()
        for i in range(len(words)):
            for lhs, log_prob in preterm[words[i]]:
                best_score["{}({}, {})".format(lhs, i, i+1)] = log_prob

        for j in range(2, len(words)+1):
            for i in range(j-2, -1, -1):
                for k in range(i+1, j):
                    for sym, lsym, rsym, logprob in nonterm:
                        if best_score["{}({}, {})".format(lsym, i, k)] > float('-inf') and best_score["{}({}, {})".format(rsym, k, j)] > float("-inf"):
                            my_lp = best_score["{}({}, {})".format(lsym, i, k)] + best_score["{}({}, {})".format(rsym, k, j)] + logprob
                        if my_lp > best_score["{}({}, {})".format(sym, i, j)]:
                            best_score["{}({}, {})".format(sym, i, j)] = my_lp
                            best_edge["{}({}, {})".format(sym, i, j)] = ("{}({}, {})".format(lsym, i, k), "{}({}, {})".format(rsym, k, j))
        print(PRINT("S(0, {})".format(len(words))))
