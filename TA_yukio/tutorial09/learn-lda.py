# python learn-lda.py ../../data/wiki-en-documents.word my_topics.txt 100 10 0.01 0.01

from collections import defaultdict
import sys
import random
import math

def add_counts(word, topic, docid, amount, xcounts, ycounts):
    xcounts[topic] += amount
    xcounts["{}|{}".format(word, topic)] += amount

    ycounts[docid] += amount
    ycounts["{}|{}".format(topic, docid)] += amount

    if xcounts[topic] < 0 or xcounts["{}|{}".format(word, topic)] < 0 or ycounts[docid] < 0 or ycounts["{}|{}".format(topic, docid)] < 0:
        print("Counts Error (value < 0)")
        sys.exit()

def sample_one(probs):
    z = sum(probs)
    remaining = random.random() * z
    for i in range(len(probs)):
        remaining -= probs[i]
        if remaining <= 0:
            return i
    print("Sampling Error (remaining > 0)")
    sys.exit()

def learn_lda(fin_path, fout_path, num_topics, epoch, alpha, beta):
    xcorpus = list()
    ycorpus = list()
    xcounts = defaultdict(lambda: 0.0)
    ycounts = defaultdict(lambda: 0.0)
    vocabulary = defaultdict(lambda: 0)
    with open(fin_path, "r") as fin:
        for line in fin:
            docid = len(xcorpus)
            words = line.strip("\n").split(" ")
            topics = list()
            for word in words:
                topic = random.randrange(0, num_topics)
                topics.append(topic)
                add_counts(word, topic, docid, 1, xcounts, ycounts)
                vocabulary[word] = 1
            xcorpus.append(words)
            ycorpus.append(topics)
    N_x = len(vocabulary)

    for l in range(epoch):
        ll = 0
        for i in range(len(xcorpus)):
            for j in range(len(xcorpus[i])):
                x = xcorpus[i][j]
                y = ycorpus[i][j]
                add_counts(x, y, i, -1, xcounts, ycounts)
                probs = list()
                for k in range(num_topics):
                    prob = ((xcounts["{}|{}".format(x, k)] + alpha) / (xcounts[k] + alpha * N_x)) * ((ycounts["{}|{}".format(k, i)] + beta) / (ycounts[i] + beta * num_topics))
                    probs.append(prob)
                new_y = sample_one(probs)
                ll += math.log(probs[new_y])
                add_counts(x, new_y, i, 1, xcounts, ycounts)
                ycorpus[i][j] = new_y
        print("epoch:{}\tll:{}".format(l + 1, ll))

    with open(fout_path, "w") as fout:
        for words, topics in zip(xcorpus, ycorpus):
            wordtopics = list()
            for word, topic in zip(words, topics):
                wordtopics.append("{}_{}".format(word, topic))
            fout.write(" ".join(wordtopics))
            fout.write("\n\n")

    with open(fout_path + ".topics", "w") as fout:
        for words, topics in zip(xcorpus, ycorpus):
            topicwords = defaultdict(list)
            for word, topic in zip(words, topics):
                topicwords[topic].append(word)
            for i, topicword in sorted(topicwords.items()):
                fout.write("topic{}".format(i))
                fout.write("\n")
                fout.write(", ".join(topicword))
                fout.write("\n")
            fout.write("\n\n")

if __name__ == "__main__":
    learn_lda(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]))
