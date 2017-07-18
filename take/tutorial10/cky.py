from collections import defaultdict
from pprint import pprint
import sys
from math import log

TESTRUN = True

try:
    sys.argv[1] == 'go'
    TESTRUN = False
except:
    pass

if TESTRUN:
    grammar_file = '../../test/08-grammar.txt'
    input_file = '../../test/08-input.txt'
else:
    grammar_file = '../../data/wiki-en-test.grammar'
    input_file = '../../data/wiki-en-short.tok'

nonterm = list()
preterm = defaultdict(list)
# preterm = defaultdict(lambda:tuple())

with open(grammar_file) as f:
    for rule in f:
        lhs, rhs, prob = rule.strip().split('\t')
        rhs_sym = rhs.split(' ')
        # print(lhs, rhs_sym, prob)
        if len(rhs_sym) == 1: # 前終端記号
            preterm[rhs].append((lhs, log(float(prob))))
        else:
            nonterm.append((lhs, rhs_sym[0], rhs_sym[1], log(float(prob))))

BRA = '('
CKET = ')'
SPC = ' '


def print_s(_sym_ij, w: list):
    _sym, _i, _j = _sym_ij
    # pprint(best_edge)
    if _sym_ij in best_edge:
        return BRA + _sym + SPC + print_s(best_edge[_sym_ij][0], w) + SPC + print_s(best_edge[_sym_ij][1], w) + CKET
    else:
        return BRA + _sym + SPC + w[int(_i)] + CKET


# float_max = -sys.float_info.max
# float_max = sys.float_info.min #これ > 0 だった・・・罠ﾜﾛｽ
float_max = float('-inf')

with open(input_file) as f, open('ans', 'w') as ansout:
    for line in f:
        best_score = defaultdict(lambda: float_max)
        best_edge = defaultdict(tuple)
        words = line.strip().split()

        for i in range(len(words)):
            for _lhs, log_prob in preterm[words[i]]:
                if log_prob > float_max:
                    best_score[_lhs, i, i+1] = log_prob

        for j in range(2, len(words)+1):
            for i in range(j-2, -1, -1):
                # print('{}, {}'.format( i+1, j-1))
                # print('{}'.format(list(range(i+1, j))))
                for k in range(i+1, j):
                    for sym, lsym, rsym, logprob in nonterm:
                        # print(best_score)
                        # print(best_score[rsym, k, j] )
                        if best_score[lsym, i, k] > float_max and best_score[rsym, k, j] > float_max:
                            # print('{}, {}'.format(best_score[lsym, i, k], best_score[rsym, k, j]))
                            mylp = best_score[lsym, i, k] + best_score[rsym, k, j] + logprob
                            if mylp > best_score[sym, i, j]:
                                best_score[sym, i, j] = mylp
                                best_edge[sym, i, j] = (lsym, i, k), (rsym, k, j)
                                # print(best_edge)


        # print(best_score)
        # print(best_edge)

        if TESTRUN:
            print(print_s(('S', 0, len(words)), words), file=sys.stdout)
            import subprocess
            subprocess.call('echo "\n Answer of TestCase"', shell=True)
            subprocess.call('cat ../../test/08-output.txt', shell=True)
        else:
            print(print_s(('S', 0, len(words)), words), file=ansout)
