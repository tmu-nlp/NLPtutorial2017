import collections
import math
import train_bigram

def witten_bell(data):
    word_u = collections.defaultdict(lambda: [])
    word_c = collections.defaultdict(lambda: 0)
    rate_2 = collections.defaultdict(lambda: 0)
    for line in data:
        line = '<s> ' + line
        line = line.lower().strip()
        bigram = train_bigram.ngram(line.split(), 2)
        for word_1, word_2 in bigram:
            word_c[word_1] += 1
            word_u[word_1].append(word_1 + ' ' + word_2)
    for key in word_u.keys():
        word_u[key] = len(set(word_u[key]))
    for key in word_u.keys():
        rate_2[key] = 1 - (word_u[key] / (word_u[key] + word_c[key]))

    return rate_2

if __name__ == "__main__":
    N = 1000000; rate_1 = 0.95;
    entropy = 0; total_count = 0;
    with open('../../data/wiki-en-train.word', 'r') as data_train:
        word_probabilities_1, word_probabilities_2 = train_bigram.train(data_train)
    with open('../../data/wiki-en-train.word', 'r') as data_train:
        rate_2 = witten_bell(data_train)
    with open('../../data/wiki-en-test.word', 'r') as data_test:
        for line in data_test:
            line = '<s> ' + line
            line = line.lower().strip()
            bigram = train_bigram.ngram(line.split(), 2)
            total_count += len(bigram) + 1
            for word_1, word_2 in bigram:
                P1 = rate_1*word_probabilities_1[word_2] + (1 - rate_1)/N
                P2 = rate_2[word_1]*word_probabilities_2[word_1 + ' ' + word_2] + (1 - rate_2[word_1])*P1
                entropy += - math.log2(P2)
    entropy /= total_count
    print('エントロピー：{}'.format(entropy))
