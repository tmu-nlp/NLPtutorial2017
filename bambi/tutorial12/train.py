import math
import pickle
from collections import defaultdict

def create_features(X, Y):
    phi = defaultdict(int)
    for i in range(len(Y)+1):
        first_tag = "<s>" if i == 0 else Y[i-1]
        next_tag = "</s>" if i == len(Y) else Y[i]
        phi.update(CREATE_TRANS(first_tag,next_tag))
    for i in range(len(Y)):
        phi.update(CREATE_EMIT(Y[i],X[i]))
    return phi

def key(*names, delimiter=" "):
    return delimiter.join(map(str,names))

def CREATE_TRANS(prev,next_tag):
    t = defaultdict(float)
    t["T,"+str(prev)+","+str(next_tag)] = 1
    return t

def CREATE_EMIT(next_tag, words_i):
    e = defaultdict(float)
    e["E,"+str(next_tag)+","+str(words_i)] = 1
    e["CAPS,"+ next_tag] = 1
    return e

def cal_prob(w,f):
    score = 0
    for k, v in f.items():
        score += v * w[k]
    return score

def HMM_VITERBI(w,words,transition,possible_tags):
    l = len(words)
    best_score = {}
    best_edge = {}
    best_score["0 <s>"] = 0 # Start with <s>
    best_edge["0 <s>"] = None

    for i in range(0, len(words)):
        for prev in possible_tags.keys():
            for next_tag in possible_tags.keys():
                if key(i,prev) in best_score and key(prev,next_tag) in transition:
                    T = cal_prob(w, CREATE_TRANS(prev,next_tag))
                    E = cal_prob(w, CREATE_EMIT(next_tag,words[i]))
                    score = best_score[key(i,prev)] + T + E
                    next_score_key = "{} {}".format(i+1,next_tag)
                    if next_score_key not in best_score or best_score[next_score_key] < score:
                        best_score[next_score_key] = score
                        best_edge[next_score_key] = key(i,prev)

    for prev in possible_tags.keys():
        score_key = "{} {}".format(l, prev)
        trans_key = "{} </s>".format(prev)
        next_score_key = "{} </s>".format(l+1)
        if score_key in best_score and trans_key in transition:
            T = cal_prob(w, CREATE_TRANS(prev,next_tag))
            score = best_score[score_key] + T
            if next_score_key not in best_score or best_score[next_score_key] < score:
                best_score[next_score_key] = score
                best_edge[next_score_key] = score_key
    tags = []
    next_edge = best_edge["{} </s>".format(l+1)]
    print(next_edge)
    while next_edge != "0 <s>":
        position, tag = next_edge.split()
        tags.append(tag)
        next_edge = best_edge[next_edge]
    tags.reverse()
    return tags

def update_weight(w, prime,hat):
    k_prime = list(prime.keys())
    k_hat = list(hat.keys())
    keys = set(k_prime + k_hat)
    for k in keys:
        w[k] += prime[k] - hat[k]
    return w

if __name__ == '__main__':
    w = defaultdict(int)
    l = 2
    file = "../../data/wiki-en-train.norm_pos"
    #file = "../test/05-train-input.txt"
    emit = defaultdict(int)
    transition = defaultdict(int)
    context = defaultdict(int)
    possible_tags = defaultdict(int)

    for line in open(file):
        line = line.strip("\n")# clean line
        previous = "<s>"
        possible_tags[previous] = 1
        wordtags = line.split(" ")
        for wordtag in wordtags:
            word, tag = wordtag.split("_")
            transition["{} {}".format(previous,tag)] += 1
            possible_tags[tag] += 1
            emit["{} {}".format(tag,word)] += 1
            previous = tag
        transition["{} </s>".format(previous)] += 1
    possible_tags["</s>"] += 1

    for i in range(l):
        for line in open(file):
            X = []
            Y_prime = []
            for wordtag in line.strip("\n").split(" "):
                word, tag = wordtag.split("_")
                X.append(word)
                Y_prime.append(tag)

            Y_hat = HMM_VITERBI(w,X,transition,possible_tags)
            phi_prime = create_features(X, Y_prime)
            phi_hat = create_features(X, Y_hat)
            w = update_weight(w,phi_prime,phi_hat)

    #print(w)
    pickle.dump((w,possible_tags,transition),open("model.pickle","wb"))
    print("finished")
