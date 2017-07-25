from collections import defaultdict
import math
from nltk.tree import Tree



def read_grammar(data_grammar):
    nonterm = list()
    preterm = defaultdict(list)

    for rule in data_grammar:
        lhs, rhs, prob = rule.split('\t')
        rhs_symbols = rhs.split()
        if len(rhs_symbols) == 1:
            preterm[rhs].append((lhs, math.log(float(prob))))
        else:
            nonterm.append((lhs, rhs_symbols[0], rhs_symbols[1], math.log(float(prob))))

    return nonterm, preterm



def create_S_exp(sym):
    if sym not in best_edge.keys():
        return('grammar_Error')
    nodes = best_edge[sym]
    if len(nodes) == 2:
        return '({} {} {})'.format(sym[0], create_S_exp(nodes[0]), create_S_exp(nodes[1]))
    else:
        return '({} {})'.format(sym[0], nodes[0])



if __name__ == '__main__':
    with open('../../data/wiki-en-short.tok') as data_in, open('../../data/wiki-en-test.grammar') as data_grammar, open('my_answer.txt', 'w') as data_out:
        nonterm, preterm = read_grammar(data_grammar)
        inf = float("inf")

        for num, line in enumerate(data_in):
            words = line.split()

            best_score = defaultdict(lambda: -inf)
            best_edge = defaultdict(lambda: (('', 0, 0), ('', 0, 0)))
            for i in range(len(words)):
                for lhs, log_prob in preterm[words[i]]:
                    best_score[(lhs, i, i+1)] = log_prob
                    best_edge[lhs, i, i+1] = (words[i], i, i+1)
            for j in range(2, len(words)+1):
                for i in range(j-2, -1, -1):
                    for k in range(i+1, j):
                        for sym, lsym, rsym, log_prob in nonterm:
                            if best_score[(lsym, i, k)] > -inf and best_score[(rsym, k, j)] > -inf:
                                my_lp = best_score[(lsym, i, k)] + best_score[(rsym, k, j)] + log_prob
                                if my_lp > best_score[(sym, i, j)]:
                                    best_score[(sym, i, j)] = my_lp
                                    best_edge[(sym, i, j)] = ((lsym, i, k), (rsym, k, j))

            print(create_S_exp(('S', 0, len(words))), file=data_out)
            if create_S_exp(('S', 0, len(words))) != 'grammar_Error':
                t = Tree.fromstring(create_S_exp(('S', 0, len(words))))
                t.pretty_print()
                print('\n')
            else:
                print('grammar_Error\n\n')
