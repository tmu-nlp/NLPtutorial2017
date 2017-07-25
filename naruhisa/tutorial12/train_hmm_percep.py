from collections import defaultdict
import random
import pickle

def HMM_VITERBI(w, X):
    best_edge = defaultdict(None)
    best_score = defaultdict(int)
    best_score['0 <s>'] = 0
    best_edge['0 <s>'] = None
    l = len(X)
    for i in range(l):
        for Prev in possible_tags:
            for Next in possible_tags:
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

def Create_Feature(x, y):
    phi = defaultdict(int)
    for i in range(len(y) + 1):
        if i == 0:
            first_tag = '<s>'
        else:
            first_tag = y[i-1]
        if i == len(y):
            next_tag = '</s>'
        else:
            next_tag = y[i]
        phi[first_tag + ' ' + next_tag] += 1
#        Create_T(first_tag, next_tag)
    for i in range(len(y)):
        phi[x[i] + ' ' + y[i]] += 1
#        Create_E(y[i], x[i])
#    print(phi)
    return phi

def Create_E(y, x):
    phi[y + ' ' + x] += 1

def Create_T(Prev, Next):
    phi[Prev + ' ' + Next] += 1

if __name__ == '__main__':
    total = 0
    count = 0
    data = list()
    w = defaultdict(lambda:random.random() * .1)
    l_ = 5
    possible_tags = set()
    test_input = '../../test/05-train-input.txt'
    train_input = '../../data/wiki-en-train.norm_pos'
    with open(train_input) as f:
        for line in f:
            words = line.strip().split()
            X = list()
            Y_prime = list()
#            X = ['<s>']
#            Y_prime = ['<s>']
            for wordtag in words:
                word, tag = wordtag.split('_')
                X .append(word)
                Y_prime.append(tag)
                possible_tags.add(tag)
            X.append('</s>')
#            Y_prime.append('</s>')
            data.append([X, Y_prime])
        possible_tags.add('<s>')
        possible_tags.add('</s>')
        possible_tags = list(possible_tags)
        for epoch in range(l_):
            print('epoch:', epoch+1)
            random.shuffle(data)
            for X, Y_prime in data:
                total = 0
                count = 0
                phi = defaultdict(int)
                Y_hat = HMM_VITERBI(w, X)
                phi_prime = Create_Feature(X, Y_prime)
                phi_hat = Create_Feature(X, Y_hat)
                for k, v in phi_hat.items():
#                    print(k, v)
                    w[k] -= v
                for k, v in phi_prime.items():
                    w[k] += v
#                for y_p, y_h in zip(Y_prime, Y_hat):
#                    if y_p == y_h:
#                        count += 1
#                    else:
#                        print(y_p, y_h)
#                    total += 1
#                print(count/total)
    with open('weight.pkl', 'wb') as o_f:
        pickle.dump(dict(w), o_f)
