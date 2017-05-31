from collections import defaultdict
import sys

emit = defaultdict(int)
transition = defaultdict(int)
context = defaultdict(int)
with open('model.txt', 'w') as model_file:
    with open(sys.argv[1]) as learn_file:
        for line in learn_file:
            previous = '<s>'
            context[previous] += 1
            wordtags = line.split()
            for wordtag in wordtags:
                word, tag = wordtag.split('_')
                transition[previous+' '+tag] += 1
                context[tag] += 1
                emit[tag+' '+word] += 1
                previous = tag
            transition[previous+' </s>'] += 1
            print(previous)
        for key, value in transition.items():
            previous, word = key.split()
            model_file.write('T {} {}\n'.format(key,value/context[previous]))
        for key, value in emit.items():
            tag, word = key.split()
            model_file.write('E {} {}\n'.format(key,value/context[tag]))

