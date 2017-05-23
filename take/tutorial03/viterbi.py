import sys
from math import log
from collections import defaultdict

# for logging
from logging import getLogger, DEBUG, INFO, ERROR, StreamHandler
logger = getLogger('hoge')
handler = StreamHandler()
handler.setLevel(DEBUG)
# logger.setLevel(ERROR)
logger.setLevel(INFO)
logger.addHandler(handler)

# dev params
do_test = False
flag_gen_ans = True
do_print_report = flag_gen_ans and not do_test
if do_test:
    # testset
    model = '../../test/04-model.txt'
    inputfile = '../../test/04-input.txt'
    outfile = 'test.ans'
else:
    model = './wiki-ja-train.word.unimodel'
    inputfile = '../../data/wiki-ja-test.txt'
    outfile = 'wiki.ans'

# load model
model_defdict = defaultdict(float)
with open(model) as f:
    for w in f:
        if do_test:
            #testcase はタブ区切り
            model_defdict[w.split('\t')[0]] = w.split('\t')[1].strip()
        else:
            #自分でつくったのは、スペース区切り
            model_defdict[w.split(' ')[0]] = w.split(' ')[1].strip()

logger.debug('model defdict:{}'.format(model_defdict))

# set h-params
lambda_coef = 0.95
lambda_unk = 1. - lambda_coef
V = 10**6

best_edge = defaultdict(tuple)
best_score = defaultdict(float)
with open(inputfile) as f, open(outfile,"w") as out:
    for line in f:
        # Step forward
        logger.debug('\n---Start Line---')
        logger.debug("line length : {} : {}".format(len(line), len(line.strip('\n'))))
        best_edge[0] = None
        best_score[0] = 0
        stripped_line = line.strip('\n') # lenに影響するので改行コード削除
        logger.debug(stripped_line)
        for word_end in range(1, len(stripped_line) + 1): # 最後の文字まで走査するので+1
            logger.debug('---- line length : {}'.format(len(line.strip())))
            logger.debug("word_end:{}".format(word_end))
            logger.debug("line -> {} ; line length{}".format(line.strip(),len(line.strip())))
            best_score[word_end] = float(sys.maxsize)
            for word_begin in range(word_end):#頭の文字から前向きにword_endまで全ノード走査
                logger.debug("word_begin:{}, word_end:{}".format(word_begin,word_end))
                if word_begin >= word_end: # Sanitize
                    break
                word = stripped_line[word_begin:word_end]
                logger.debug("checking -> {}".format(word))
                prob = lambda_unk/V #未知語分の確率は最初に与えておく
                if word in model_defdict.keys() or len(word) == 1:#モデルにない2文字以上は考えない
                    prob += lambda_coef *  float(model_defdict[word])
                    logger.debug("prob: {} ".format(prob))
                    myscore = best_score[word_begin] - log(float(prob)) # いまの分割ポイントまでのスコアを現在の最小値と比較
                    if myscore < best_score[word_end]:#最小値からさらに小さければ更新
                        best_score[word_end] = myscore
                        best_edge[word_end] = (word_begin, word_end)
                else:
                    pass
                    #logger.debug('into else: {}'.format(word))

        for k,v in best_edge.items():
            logger.debug("best_edge {}:{}".format(k,v))

        # Step backward
        words = [] # 最短経路の格納用リストで、最後に反転する
        next_edge = best_edge[len(best_edge)-1] # 構築した最小パスの最後尾,elemはtapleで分割情報が格納されている
        logger.debug("nextedge-> {}".format(next_edge))
        while next_edge is not None: # 全エッジをたどる
            logger.debug("line: {}".format(line.strip()))
            word = stripped_line[next_edge[0]:next_edge[1]] # 現在走査中の分割単語を得る
            # if len(word) > 0 and word != '\n': # sanitize、不要なチェックである.
            #     logger.debug("word: {}".format( word))
            words.append(word) # 最短経路リストに追加する
            next_edge = best_edge[next_edge[0]]
        words.reverse()
        logger.debug(words)
        logger.info(' '.join(words))
        # emit to file
        if flag_gen_ans:
            out.write(' '.join(words)+'\n')

# if flag_gen_ans and not do_test:
if do_print_report:
    print('\n ------------ REPORT ------------\n')
    import subprocess
    ret = subprocess.call('../../script/gradews.pl ../../data/wiki-ja-test.word ' + outfile, shell=True)
