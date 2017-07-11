from collections import defaultdict
import numpy as np

def PRINT(S):
	if S in best_edge:
		return '(' + S.split(',')[0] + ' ' + PRINT(best_edge[S][0]) + ' ' + PRINT(best_edge[S][1]) + ')'
	else:
		return '(' + S.split(',')[0] + ' ' + words[int(S.split(',')[1])] + ')'

def Numberize(sym, i, j):
	return sym + ',' + str(i) + ',' + str(j)


if __name__ == '__main__':
	noterm = list()
	preterm = defaultdict(list)
	inf = 10 ** 6
	best_score = defaultdict(lambda: inf)
	best_edge = defaultdict(list)
	test_grammer = '../../test/08-grammar.txt'
	test_grammer_ex = '../../data/wiki-en-test.grammar'

	with open(test_grammer_ex, 'r') as g_f:
		for line in g_f:
			lhs, rhs, prob = line.strip().split('\t')
			prob = float(prob)
			rhs_symbols = rhs.split(' ')
			if len(rhs_symbols) == 1:
				preterm[rhs_symbols[0]].append((lhs, np.log(prob)))
			else:
				noterm.append([lhs, rhs_symbols[0], rhs_symbols[1], np.log(prob)])

	test_input = '../../test/08-input.txt'
	test_input_ex = '../../data/wiki-en-short.tok'

	with open(test_input_ex, 'r') as i_f:
		for line in i_f:
			words = line.split()
			print(line)
			for i in range(len(words)):
				for lhs, log_prob in preterm[words[i]]:
					best_score[Numberize(lhs, i, i+1)] = log_prob

			for j in range(2, len(words)+1):
				for i in reversed(range(j-1)):
					for k in range(i+1, j):
						for sym, lsym, rsym, logprob in noterm:
							if best_score[Numberize(lsym, i, k)] > -inf and best_score[Numberize(sym, k, j)] > -inf:
								my_lp = best_score[Numberize(lsym, i, k)] + best_score[Numberize(rsym, k, j)] + logprob
								if my_lp > best_score[Numberize(sym, i, j)]:
									best_score[Numberize(sym, i, j)] = my_lp
									best_edge[Numberize(sym, i, j)] = (Numberize(lsym, i, k), Numberize(rsym, k, j))
#			print(best_edge)
#			print(preterm)
			print(PRINT(Numberize('S', 0, len(words))))
