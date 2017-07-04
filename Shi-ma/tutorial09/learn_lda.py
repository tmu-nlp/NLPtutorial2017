from collections import defaultdict
import random
import math
import pprint



def SAMPLEONE(probs):
    z = sum(probs)
    remaining = random.uniform(0, z)
    for i in range(len(probs)):
        remaining -= probs[i]
        if remaining <= 0:
            return i
        if i == (len(probs)-1):
            print ('SAMPLEONE_Error')
            exit()



def ADDCOUNTS(word, topic, docid, amount):
    x_counts[str(topic)] += amount
    x_counts[word + '|' + str(topic)] += amount
    if x_counts[str(topic)] < 0 or x_counts[word + '|' + str(topic)] < 0:
        print('ADDCOUNTS_x_Error')
        exit()

    y_counts[str(docid)] += amount
    y_counts[str(topic) + '|' + str(docid)] += amount
    if y_counts[str(docid)] < 0 or y_counts[str(topic) + '|' + str(docid)] < 0:
        print('ADDCOUNTS_y_Error')
        exit()



if __name__ == '__main__':
    path_data_in = '../../data/wiki-en-documents.word'
    # path_data_in = '../../test/07-train.txt'
    num_topics = 2
    num_epochs = 10
    num_words = 0
    alpha = 0.01
    beta = 0.01
    x_courpus = list()
    y_courpus = list()
    x_counts = defaultdict(lambda: 0)
    y_counts = defaultdict(lambda: 0);

    with open(path_data_in, 'r') as data_in:
        for line in data_in:
            docid = len(x_courpus)
            words = line.strip().split()
            topics = list()
            for word in words:
                num_words += 1
                topic = random.randint(0, num_topics - 1)
                topics.append(topic)
                ADDCOUNTS(word, topic, docid, 1)
            x_courpus.append(words)
            y_courpus.append(topics)

    for num_epoch in range(num_epochs):
        print('num_epoch = {}'.format(num_epoch + 1))
        ll = 0
        for i in range(len(x_courpus)):
            for j in range(len(x_courpus[i])):
                x = x_courpus[i][j]
                y = y_courpus[i][j]
                ADDCOUNTS(x, y, i, -1)
                probs = list()
                for k in range(num_topics):
                    prob_x = (x_counts[x + '|' + str(k)] + alpha) / (x_counts[str(k)] + alpha*num_words)
                    prob_y = (y_counts[str(k) + '|' + str(i)] + beta) / (y_counts[str(i)] + beta*num_topics)
                    probs.append(prob_x * prob_y)
                new_y = SAMPLEONE(probs)
                ll += math.log(probs[new_y])
                ADDCOUNTS(x, new_y, i, 1)
                y_courpus[i][j] = new_y
        print (ll)

    with open('result.txt', 'w') as data_out:
        print('x_counts\n', file=data_out)
        data_out.write(pprint.pformat(sorted(x_counts.items(), key=lambda x: x[0][-1])))
        print('\n\n', file=data_out)

        print('y_counts\n', file=data_out)
        data_out.write(pprint.pformat(sorted(y_counts.items(), key=lambda x: x[0])))
