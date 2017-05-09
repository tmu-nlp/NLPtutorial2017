import collections
import math
import train_unigram

if __name__ == "__main__":
    N = 1000000; rate_unk = 0.05;
    entropy = 0; total_count = 0; unk_count = 0
    word_coverage = collections.defaultdict(lambda :0)
    with open('../../data/wiki-en-train.word', 'r') as data_train:
        word_probabilities = train_unigram.train(data_train)
    with open('../../data/wiki-en-test.word', 'r') as data_test:
        for line in data_test:
            for word in line.split():
                if word == '.':
                    continue
                total_count += 1
                if word in word_probabilities.keys():
                    P = (1 - rate_unk)*word_probabilities[word] + rate_unk/N
                else:
                    unk_count += 1
                    P = rate_unk/N
                entropy += - math.log2(P)
        entropy /= total_count
        coverage = (total_count - unk_count)/(total_count)
    print('エントロピー：{}'.format(entropy))
    print('カバレッジ：{}'.format(coverage))
