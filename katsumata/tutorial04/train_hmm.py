from collections import defaultdict
emit = defaultdict(int)
transition = defaultdict(int)
context = defaultdict(int)

with open ('../../data/wiki-en-train.norm_pos', 'r') as train_file:
    for line in train_file:
        #line = line.strip()
        print (line)
        previous = '<s>'
        context[previous] += 1
        wordtags = line.split(' ')
        for wordtag in wordtags:
            word = wordtag.split('_')[0]
            tag = wordtag.split('_')[1]
            transition[previous+' '+ tag] += 1 #遷移を数え上げる
            context[tag] += 1 #文脈を数え上げる
            emit[tag +' '+ word] += 1 #生成を数え上げる
            previous = tag
        transition[previous+' </s>'] += 1   
        #遷移確率を出力
for key, value in transition.items():
    previous = key.split(' ')[0]
    word = key.split(' ')[1]
    print('T {} {}'.format(key, value/context[previous]))
for key, value in emit.items():
    tag = key.split(' ')[0]
    word = key.split(' ')[1]
    print('E {} {}'.format(key, value/context[tag]))
