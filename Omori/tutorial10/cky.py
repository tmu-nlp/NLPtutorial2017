from collections import defaultdict
import math

def read_grammer():
    preterm = defaultdict(list)
    nonterm = list()

    with open('../../data/wiki-en-test.grammar') as f:
        for line in f:
            lhs, rhs, prob = line.strip().split('\t')
            rhs_symbols = rhs.split(' ')
            if len(rhs_symbols) == 1:
                preterm[rhs].append((lhs, math.log(float(prob), 2)))
            else:
                nonterm.append((lhs, rhs_symbols[0], rhs_symbols[1], math.log(float(prob), 2)))
   
    return preterm, nonterm

def print_tree(sym, best_edge, words):
    if sym in best_edge:
        s = '({} {} {})'.format(sym.split()[0], print_tree(best_edge[sym][0], best_edge, words), print_tree(best_edge[sym][1], best_edge, words))
    else:
        s = '({} {})'.format(sym.split()[0], words[int(sym.split()[1])])

    return s

def main():
    preterm, nonterm = read_grammer()

    with open('../../data/wiki-en-short.tok') as f:
        for line in f:
            words = line.strip().split()
            best_score = defaultdict(lambda: -10000.0)
            best_edge = defaultdict(tuple)
            for i, word in enumerate(words):
                for lhs, prob in preterm[word]:
                    best_score[lhs+' {} {}'.format(i, i+1)] = prob    
            
            for j in range(2, len(words)+1):
                for i in range(j-2, -1, -1):
                    for k in range(i+1, j):
                        for sym, lsym, rsym, prob in nonterm:
                            if (lsym+' {} {}'.format(i, k) in best_score) and (rsym+' {} {}'.format(k, j) in best_score):
                                my_lp = best_score[lsym+' {} {}'.format(i, k)]+best_score[rsym+' {} {}'.format(k, j)]+prob
                                if my_lp > best_score[sym+' {} {}'.format(i, j)]:
                                    best_score[sym+' {} {}'.format(i, j)] = my_lp
                                    best_edge[sym+' {} {}'.format(i, j)] = (lsym+' {} {}'.format(i, k), rsym+' {} {}'.format(k, j))
            

            print(print_tree('S {} {}'.format(0, len(words)), best_edge, words))


if __name__ == "__main__":
    main()
