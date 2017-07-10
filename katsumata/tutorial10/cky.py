import math
from collections import defaultdict

#lhs\trhs\tprob形式
def readGrammer(file_name):
    nonterm = list() #非終端記号
    preterm = defaultdict(list) #pre[右]=[(left, prob)]
    with open(file_name) as g_f:
        for line in g_f:
            lhs, rhs, prob = line.split('\t')
            rhs = rhs.split(' ')
            if len(rhs) == 1:
                preterm[rhs[0]].append((lhs, -math.log(float(prob))))
            else:
                nonterm.append((lhs, rhs[0], rhs[1], -math.log(float(prob)))) 
    return nonterm, preterm

def makeKey(symbol, x, y):
    return '{}\t{} {}'.format(symbol, x, y)

def RoutinePrint(sym):
    if sym in best_edge:
        return '({} {} {})'.format(sym.split()[0], RoutinePrint(best_edge[sym][0]), RoutinePrint(best_edge[sym][1]))
    else:
        return '({} {})'.format(sym.split()[0], words[int(sym.split()[1])])

if __name__ == '__main__':
    grammer_file = '../../data/wiki-en-test.grammar'
    #grammer_file = '../../test/08-grammar.txt'
    input_file = '../../data/wiki-en-short.tok'
    #input_file = '../../test/08-input.txt'
    output_file = 'my_answer.cky'
    #output_file = 'test_my_answer.cky'
    INF = 10 ** 5
    nonterm, preterm = readGrammer(grammer_file)
    #前終端記号を追加
    """
    注意事項
    symbolがどこの部分のものかもkeyに付与させる必要あり
    同一symbolが上書きされてしまう
    今回は'sym<tab>x<space>y'をkeyとした
    """
    with open(input_file) as i_f, open(output_file, 'w') as o_f:
        for line in i_f:
            best_score = defaultdict(lambda : INF) #引数sym 値:最大対数確率
            best_edge = dict() #引数sym 値(lsym, rsym)(1個前)
            words = line.split()
            #終端記号まわり
            for i in range(len(words)):
                for lhs, log_prob in preterm[words[i]]:
                    best_score[makeKey(lhs,i,i+1)] = log_prob
            for j in range(2, len(words)+1):
                for i in reversed(range(j-1)):
                    for k in range(i+1, j):
                        for sym, lsym, rsym, logprob in nonterm:
                            if best_score[makeKey(lsym,i,k)] < INF and best_score[makeKey(rsym,k,j)] < INF:
                                my_lp = best_score[makeKey(lsym,i,k)] + best_score[makeKey(rsym,k,j)] + logprob
                                if my_lp < best_score[makeKey(sym,i,j)]:
                                    best_score[makeKey(sym,i,j)] = my_lp
                                    best_edge[makeKey(sym,i,j)] = (makeKey(lsym,i,k), makeKey(rsym,k,j))
            o_f.write(RoutinePrint(makeKey('S',0,len(words)))+'\n')
