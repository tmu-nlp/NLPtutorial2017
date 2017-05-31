from collections import defaultdict

#def map_make(ins_f):
emit = defaultdict(int)
transition = defaultdict(int)
context = defaultdict(int)
#word_1 = defaultdict(int)
with open('../../data/wiki-en-train.norm_pos','r') as f:
    for line in f:
        line = line.strip()
#        print(line)
#        line = line.strip('\n')
        previous = "<s>"
        context[previous] += 1
        wordtags = line.split(' ')
        for wordtag in wordtags:
            word0_tag1 = wordtag.split('_')
            word = word0_tag1[0]
            tag = word0_tag1[1]
            transition[previous + ' ' + tag] += 1
            context[tag] += 1
            emit[tag + ' ' + word] += 1
            previous = tag
        transition[previous + ' </s>'] += 1
#return(transition,context)
#"""

#    sumple = map_make(f)
    for key,value in transition.items():
        p0_w1 = key.split(' ')
        previous_1 = p0_w1[0]
#        print(key)
#        if p0_w1[1]:
        word_1 = p0_w1[1]
#        print(value)
#        print(context[previous_1])
        print('{} {} {}'.format('T',key,float(value)/float(context[previous_1])))

    for key,value in emit.items():
        t0_w1 = key.split(' ')
        tag_1 = t0_w1[0]
        word_1 = t0_w1[1]
        print('{} {} {}'.format('E',key,float(value)/float(context[tag_1])))
