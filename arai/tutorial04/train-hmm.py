from collections import defaultdict

with open('../../data/wiki-en-train.norm_pos') as text:
    emit = defaultdict(int)
    transition = defaultdict(int)
    context = defaultdict(int)
    for line in text:
        previous = "<s>"
        context[previous] += 1
        for wordtag in line:
            wordtags = line.split()
            for wordtag in wordtags:
                word, tag = wordtag.split('_')   
                transition[previous + " "+tag] += 1
                context[tag] += 1
                emit[tag + " " + word] += 1
                previous = tag
            transition[previous + " </s>"] += 1

    for key, value in transition.items():
        previous, word = key.split()
        print("T " + key + " " + str(value/context[previous]))
    for key, value in emit.items():
        previous, word = key.split()
        print("E " + key + " " + str(value/context[previous]))

    
