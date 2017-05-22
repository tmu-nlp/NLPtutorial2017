import sys
from math import log
from collections import defaultdict

model_defdict = defaultdict(float)
do_test = False
flag_gen_ans = True

if do_test:
    # testset
    model = '../../test/04-model.txt'
    inputfile = '../../test/04-input.txt'
    outfile = 'test.ans'
else:
    model = './wiki-ja-train.word.unimodel'
    inputfile = '../../data/wiki-ja-test.txt'
    outfile = 'wiki.ans'

with open(model) as f:
    for w in f:
        if do_test:
            #testcase はタブ区切り
            model_defdict[w.split('\t')[0]] = w.split('\t')[1].strip()
        else:
            #自分でつくったのは、スペース区切り
            model_defdict[w.split(' ')[0]] = w.split(' ')[1].strip()

# print('model defdict',model_defdict)

lambda_coef = 0.95
lambda_unk = 1. - lambda_coef
V = 10**6

best_edge = defaultdict(tuple)
best_score = defaultdict(float)
with open(inputfile) as f, open(outfile,"w") as out:
    for line in f:
        striped_line = line.strip('\n')
        # print('\n---Start Line---',end='')
        # print("line length : {} : {}".format(len(line), len(line.strip('\n'))))
        # print(striped_line)
        best_edge[0] = None
        best_score[0] = 0
        for word_end in range(1,1+len(striped_line)):
            # print('---- line length : {}'.format(len(line.strip())))
            # print("word_end:{}".format(word_end))
            # print("line -> {} ; line length{}".format(line.strip(),len(line.strip())))
            best_score[word_end] = sys.maxsize
            for word_begin in range(word_end):
                # print("word_begin:{}, word_end:{}".format(word_begin,word_end))
                if word_begin >= word_end:
                    break
                word = striped_line[word_begin:word_end]
                # print("checking -> ",word)
                prob = lambda_unk/V
                if word in model_defdict.keys() or len(word) == 1:
                    prob += lambda_coef *  float(model_defdict[word])
                    # print("prob: ",prob)
                    myscore = best_score[word_begin] - log(float(prob))
                    if myscore < best_score[word_end]:
                        best_score[word_end] = myscore
                        best_edge[word_end] = (word_begin, word_end)
                else:
                    pass
                    # print('into else: ', word)

        # for k,v in best_edge.items():
        #     print("best_edge {}:{}".format(k,v))

        words = []
        next_edge = best_edge[len(best_edge)-1]
        # print("nextedge->",next_edge)
        while next_edge != None:
            # print("line: ", line.strip())
            word = striped_line[next_edge[0]:next_edge[1]]
            if len(word) > 0 and word != '\n':
                # print("word: ",word)
                words.append(word)
            next_edge = best_edge[next_edge[0]]
        words.reverse()
        # print(words)
        # print(words)
        # print(' '.join(words), end='')
        if flag_gen_ans:
            out.write(' '.join(words)+'\n')

print('\n ------- RESULT -------\n')
if flag_gen_ans and not do_test:
    import subprocess
    ret = subprocess.call('../../script/gradews.pl ../../data/wiki-ja-test.word ' + outfile, shell=True)
    print(ret)
