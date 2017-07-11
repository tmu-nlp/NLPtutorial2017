import random
from collections import Counter
import math

xcorpus = []
ycorpus = []
xcounts = Counter()
ycounts = Counter()
wordcounts = Counter()

learn_file = '07-train.txt'
#learn_file = '../../data/wiki-en-documents.word'
NUM_TOPICS = 2
iter_n = 1000
alpha = 0.1
beta = 0.1

def AddCounts(word, topic, docid, amount):
    xcounts[topic] += amount
    xcounts[word,topic] += amount
    if xcounts[topic] <0 or xcounts[word,topic] <0:
        print('xcounts - error')
        exit()

    ycounts[docid] += amount
    ycounts[topic, docid] += amount
    if ycounts[docid] <0 or ycounts[topic, docid]<0:
        print('ycounts - error')
        exit()

def P_x_y(x,y):
    p = (xcounts[x,y] + alpha)/(xcounts[y] + alpha*len(wordcounts))
    #print(p)
    return p
def P_y_Y(y, Y):
    p = (ycounts[y,Y] + beta)/(ycounts[Y] + beta * NUM_TOPICS)
    #print(p)
    return p

def SampleOne(probs):
    z = sum(probs)
    #print(z)
    remaining = random.uniform(0,z)
    for i in range(len(probs)):
        remaining -= probs[i]
        if remaining <= 0:
            return i
        if i == (len(probs)-1):
            print('SampleOneでおかしい')
            exit()

if __name__ == "__main__":

    with open(learn_file, 'r') as ff:
        for line in ff:
            docid = len(xcorpus)
            words = line.strip().split()
            topics = []
            for word in words:
                topic = random.randint(0,NUM_TOPICS-1)
                topics.append(topic)
                AddCounts(word,topic,docid,1)
                wordcounts[word] += 1

            xcorpus.append(words)
            ycorpus.append(topics)



    #print('X_corpus',xcorpus,'\nY_corpus',ycorpus)
    #print('X_count',xcounts,'\nY_count', ycounts)


    for ii in range(iter_n):
        ll=0
        #random.shuffle(xcorpus)
        for i in range(len(xcorpus)):
            for j in range(len(xcorpus[i])):
                x= xcorpus[i][j]
                y= ycorpus[i][j]
                AddCounts(x,y,i,-1)
                probs = []
                for k in range(NUM_TOPICS):
                    #print(P_x_y(x,k))
                    #print(P_y_Y(k,i))
                    probs.append(P_x_y(x,k) * P_y_Y(k,i))
                #print(probs)
                new_y = SampleOne(probs)
                #print(new_y)
                ll += math.log(probs[new_y])
                AddCounts(x, new_y, i, 1)
                ycorpus[i][j] = new_y
        print(ll)

    print(*xcounts,'\n', *ycounts)
    print()
    for i in range(len(xcorpus)):
        print(xcorpus[i], ycorpus[i])

    #print(*xcorpus,'\n', *ycorpus)
