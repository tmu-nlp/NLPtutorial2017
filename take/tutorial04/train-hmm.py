from collections import defaultdict
from pprint import pprint

emit = defaultdict(int)
transition = defaultdict(int)
context = defaultdict(int)

# train_input = '../../test/05-train-input.txt'
# model_file = 'test.model'

train_input = '../../data/wiki-en-train.norm_pos'
model_file = 'wiki.model'

with open(train_input, "r") as f, open(model_file, 'w') as f_out:
    for line in f:
        prev_tag = '<s>'#品詞列の開始記号
        context[prev_tag] += 1
        wordtags_list = line.strip().split(' ') #改行コードを落としてスペースで分割
        for wordtag in wordtags_list:
            word, tag = wordtag.split('_')
            emit[tag + ' ' + word] += 1 #品詞生成する単語を数え上げ
            transition[prev_tag + ' ' + tag] += 1 #品詞の遷移を数える
            context[tag] += 1 #確率計算するときの分母、tag(品詞)の総数
            # print('prevtag:{}, tag:{}'.format(prev_tag, tag))
            prev_tag = tag  # "前回のタグ"を更新
        transition[prev_tag + ' </s>'] += 1 #一文を読み終え。終端記号までの品詞の遷移としてを追加する。
    pprint(transition)
    pprint(emit)
        # print('{}, {}'.format(word,tag))
    for k, v_count in transition.items(): #transitionには、キーに品詞遷移、バリューに
        prevtag, tag = k.split(' ')
        # print('prevtag:{}, tag:{}'.format(prevtag, tag))
        # print('T {} {}'.format(k, round(v_count/context[prevtag], 6)))
        f_out.write('T {} {}\n'.format(k, round(v_count/context[prevtag], 6)))
    # print('---------------')
    for k, v_count in emit.items():#emitには、キーに"品詞->単語",バリューにその生成のカウント
        tag, word = k.split(' ')
        # print('E {} {}'.format(k, round(v_count/context[tag], 6)))
        f_out.write('E {} {}\n'.format(k, round(v_count/context[tag], 6)))
