from collections import defaultdict

emit = defaultdict(lambda: 0)
transition = defaultdict(lambda: 0)
context = defaultdict(lambda: 0)
with open('../../data/wiki-en-train.norm_pos', 'r') as f:
    for line in f:
        previous = '<s>'
        context[previous] += 1
        line = line.split()
        for wordtag in line:
            word, tag = wordtag.split('_')
            transition[previous + ' ' + tag] += 1
            context[tag] += 1
            emit[tag + ' ' + word] += 1
            previous = tag
        transition[previous + ' </s>'] += 1
        print(previous + ' </s>', transition[previous + ' </s>'])

    with open('result.txt', 'w') as fw:
        for k, v in transition.items():
            if(k != ''):
                previous = k.split(' ')[0]
                fw.write('T {}  {}\n'  .format(k, v/context[previous]))
        for k, v in emit.items():
            previous =k.split()[0]
            fw.write('E {}  {}\n' .format(k, v/context[previous]))
