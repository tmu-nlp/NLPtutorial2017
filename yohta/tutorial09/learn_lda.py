import random
from collections import defaultdict
import math

def sumple_one(probs):
    z = sum(probs)
    remaining = random.random(0,z)
    for i in range(len(probs)):
        remaining -= probs[i]
        if remaining <= 0:
            return i
        if i == len(probs) - 1:
            print('error at function sumple_one')
            exit()

def add_counts(word,topic,docid,amount):
    x_counts[topic] += amount
    x_counts[word +'|'+ str(topic)] += amount
    if x_counts[topic] < 0 or x_counts[word +'|'+ str(topic)] < 0:
        print ("error at function add_counts's x_counts")
        exit()
    y_counts[docid] += amount
    y_counts[str(topic) +'|'+ str(docid)] += amount
    if y_counts[docid] < 0 or x_counts[str(topic) +'|'+ str(docid)] < 0:
        print ("error at function add_counts's y_counts")
        exit()

num =  5 # number of topics
a = .05
b = .05
epoch = 10

if __name__ == '__main__':
    # initialize
    x_corpus = []
    y_corpus = []
    x_counts = defaultdict(lambda :0)
    y_counts = defaultdict(lambda :0)
    x_token = defaultdict(lambda :0)
    with open('../../data/wiki-en-documents.word','r') as i_f:
        for line in i_f:
            docid = len(x_corpus)
            words = line.split()
            topics = []
            for word in words:
                topic = random.randint(0,num)
                topics.append(topic)
                add_counts(word,topic,docid,1)
                x_token[word] += 1
            x_corpus.append(words)
            y_corpus.append(topics)
    # sumpling
    for ep in range(epoch):
        print('epoch : {}\n'.format(epoch))
        ll = 0
        for i in range(len(x_corpus)):
            for j in range(len(x_corpus[i])):
                probs = []
                x = x_corpus[i][j]
                y = y_corpus[i][j]
                add_counts(x,y,i,-1)
                for k in range(num):
                    prob_ = (x_counts[x +'|'+ str(k)] + a) / (x_counts[k] + a * len(x_token))
                    prob_ *= (y_counts[str(k) +'|' + str(i)] + b) / (y_counts[i] + b * num)
                    probs.append(prob_)
                new_y = sumple_one(probs)
                ll += math.log(probs[new_y])
                add_counts(x,new_y,i,1)
                y_corpus[i][j] = new_y
#        print(ll)

        for i in range(len(x_corpus)):
            for j in range(len(x_corpus[i])):
                print ('{}\t{} '.format(x_corpus[i][j], y_corpus[i][j]))
