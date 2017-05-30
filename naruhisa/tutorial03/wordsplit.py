import math

V = 1000000
l = 0.95
l_unk = 1 - l
p_uni = dict()
bestedge = list()
bestscore = list()
with open('modelfile.txt', 'r') as f3:
    for line in f3:
        line = line.split()
        p_uni[line[0]] = float(line[1])
    with open('../../data/wiki-ja-test.txt', 'r') as text:
        for line in text:
#            line = unicode(line, 'utf-8')
            bestedge = [None for i in range(len(line))]
            bestscore = [0 for i in range(len(line))]
            for word_end in range(1, len(line)):
                bestscore[word_end] = 10000000000
                for word_begin in range(word_end):
                    word = line[word_begin:word_end]
                    if(word in p_uni or len(word) == 1):
                        prob = l_unk / V
                        if(word in p_uni):
                            prob += l * p_uni[word]
                        my_score = float(bestscore[word_begin] + -math.log(prob))
                        if(my_score < bestscore[word_end]):
                            bestscore[word_end] = my_score
                            bestedge[word_end] = (word_begin, word_end)

            words = []
            next_edge = bestedge[len(bestedge) - 1]
            while(next_edge != None):
                word = line[next_edge[0]:next_edge[1]]
                words.append(word)
                next_edge = bestedge[next_edge[0]]
            words.reverse()
            print(' '.join(words).strip())
