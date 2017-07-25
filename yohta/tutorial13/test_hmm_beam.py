from collections import defaultdict
import math


V = 10**6
lambda1 = 0.1
B = 20

with open('../tutorial04/model-hmm.txt') as model_file:
    trans = defaultdict(lambda :1)
    emit = defaultdict(float)
    possible_tags = dict()
    for line in model_file:
        tag, context, word, prob = line.strip().split()
        possible_tags[context] = 1
        if tag == 'T':
            trans[context + ' ' + word] = float(prob)
        else:
            emit[context + ' ' + word] = float(prob)


with open('../../data/wiki-en-test.norm') as text:
    for line in text:
        words = line.strip().split()
        l = len(words)
        best_score = dict()
        best_edge = dict()
        tags = list()
        best_score['0 <s>'] = 0
        best_edge['0 <s>'] = None
        tags.append(['<s>'])
        for i in range(l):
            my_best = dict()
            for prev in tags[i]:
                for nex in possible_tags.keys():
                    if (str(i) + ' ' + prev) in best_score and (prev + ' ' + nex) in trans:
                        score = float(best_score[str(i) + ' ' + prev]) + float(math.log2(trans[prev + ' ' + nex])) * (-1.0) + float(math.log2(lambda1 * emit[nex + ' ' + words[i]] + (1.0 - lambda1) / V) * (-1.0))
                        if (str(i + 1) + ' ' + nex) not in best_score or best_score[str(i + 1) + ' ' +  nex] > score:
                            best_score[str(i + 1) + ' ' + nex] = score
                            best_edge[str(i + 1) + ' ' + nex] = str(i) + ' ' + prev
                            my_best[nex] = score
            tags.append([])

            for key, value in sorted(my_best.items(), key = lambda x:x[1]):
                tags[i + 1].append(key)
                if len(tags[i + 1]) == B:
                  break

        for prev in tags[i]:
            if (str(len(words)) + ' ' + prev) in best_score:
                score = float(best_score[str(len(words)) + ' ' + prev]) + float(math.log2(trans[prev + ' ' + '</s>'])) * (-1.0) + float(math.log2(emit[prev + ' ' + '</s>'] + (1.0 - lambda1) / V) * (-1.0))
                if (str(len(words) + 1) + ' ' + '</s>') not in best_score or best_score[str(len(words) + 1) + ' ' + '</s>'] > score:
                    best_score[str(len(words) + 1) + ' ' + '</s>'] = score
                    best_edge[str(len(words) + 1) + ' ' + '</s>'] = str(len(words)) + ' ' + prev
        tags = []
        next_edge = best_edge[str(l + 1)  + ' ' + '</s>']

        while next_edge != '0 <s>':
            position, tag = next_edge.split()
            tags.append(tag)
            next_edge = best_edge[next_edge]
        tags.reverse()
        print(' '.join(tags))
