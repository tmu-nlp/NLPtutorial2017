import random
import math
from collections import defaultdict

def sampleOne(probs):
    z = sum(probs)
    remaining = random.uniform(0,z)
    for i in range(len(probs)):
        remaining -= probs[i]
        if remaining <= 0:
            return i
        if i == (len(probs)-1):
            print ('sampleOneでerror')
            exit()

def addCounts(word, topic, docid, amount):
    x_counts[topic] += amount
    x_counts[word+'|'+str(topic)] += amount
    if x_counts[topic] < 0 or x_counts[word+'|'+str(topic)] < 0:
        print ('x_countsでerror')
        exit()

    y_counts[docid] += amount
    y_counts[str(topic)+'|'+str(docid)] += amount
    if y_counts[docid] < 0 or y_counts[str(topic)+'|'+str(docid)] < 0:
        print ('y_countsでerror')
        exit()

if __name__ == '__main__':
    test_file = '../../test/07-train.txt'
    learning_file = '../../data/wiki-en-documents.word'
    NUM_TOPICS = int(input('トピック数:'))
    alph = .01
    beta = .01
    l = 10
    #初期化
    x_courpus = list()
    y_courpus = list()
    x_counts = defaultdict(int)
    y_counts = defaultdict(int)
    x_token = defaultdict(int)
    with open(test_file) as i_f:
        for line in i_f:
            docid = len(x_courpus)
            words = line.split()
            topics = list()
            for word in words:
                topic = random.randint(0, NUM_TOPICS)
                topics.append(topic)
                addCounts(word, topic, docid, 1)
                x_token[word] += 1
            x_courpus.append(words)
            y_courpus.append(topics)
    #サンプリング
    for _ in range(l):
        print ('epoch {}'.format(_))
        ll = 0
        for i in range(len(x_courpus)):
            for j in range(len(x_courpus[i])):
                x = x_courpus[i][j]
                y = y_courpus[i][j]
                addCounts(x, y, i, -1)
                probs = list()
                for k in range(NUM_TOPICS):
                    probability = (x_counts[x+'|'+str(k)] + alph) / (x_counts[k] + alph*len(x_token))#p(x|y)の計算
                    probability *= (y_counts[str(k)+'|'+str(i)]+beta) / (y_counts[i] + beta*NUM_TOPICS) #p(y|Yの計算)
                    probs.append(probability)
                new_y = sampleOne(probs)
                ll += math.log(probs[new_y])
                addCounts(x, new_y, i, 1)
                y_courpus[i][j] = new_y
        print (ll)     
    """
    print ('x_counts',dict(x_counts))
    print ('y_counts',dict(y_counts))
    """
    #test
    for i in range(len(x_courpus)):
        for j in range(len(x_courpus[i])):
            x = x_courpus[i][j]
            y = y_courpus[i][j]
            print ('{}_{} '.format(x, y), end='')
        print ()    
