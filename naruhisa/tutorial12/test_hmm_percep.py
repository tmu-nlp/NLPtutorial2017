from collections import defaultdict
import pickle
import random


def HMM_VITERBI(w, X):
    best_edge = defaultdict(None)
    best_score = defaultdict(int)
    best_score['0 <s>'] = 0
    best_edge['0 <s>'] = None
    l = len(X)
    for i in range(l):
        for Prev in possible_tags:
            for Next in possible_tags:
                if Prev+' '+Next not in w:
                    w[Prev+' '+Next] = random.random()
                if X[i]+' '+Next not in w:
                    w[X[i]+' '+Next] = random.random()
                if str(i)+' '+Prev in best_score:
                    score = best_score[str(i)+' '+Prev] + w[Prev+' '+Next] + w[X[i]+' '+Next]
                    if str(i+1)+' '+Next not in best_score or score > best_score[str(i+1)+' '+Next]:
                        best_score[str(i+1)+' '+Next] = score
                        best_edge[str(i+1)+' '+Next] = str(i)+' '+Prev

    tags = []
    next_edge = best_edge[str(i+1) + ' </s>']
    while next_edge != '0 <s>':
        position, tag = next_edge.split()
        tags.append(tag)
        next_edge = best_edge[next_edge]
    tags.reverse()
    return tags

if __name__ == '__main__':
    with open('weight.pkl', 'rb') as w_f:
        w = pickle.load(w_f)
    with open('possible_tags.pkl', 'rb') as p_f:
        possible_tags = pickle.load(p_f)
    possible_tags.extend(['<s>', '</s>'])
    test_input = '../../test/05-test-input.txt'
    wiki_input = '../../data/wiki-en-test.norm'
    with open(wiki_input) as f, open('output.txt', 'w') as o_f:
        best_edge = defaultdict(None)
        best_score = defaultdict(int)
        best_score['0 <s>'] = 0
        best_edge['0 <s>'] = None
        for line in f:
            words = line.strip().split()
            words.append('</s>')
            tags = HMM_VITERBI(w, words)
            tagtag = ' '.join(tags)
            print(tagtag)
            o_f.write(tagtag + '\n')
