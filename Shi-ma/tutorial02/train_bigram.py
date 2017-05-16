import collections

def ngram(string, n):
    return list(zip(*[string[i:] for i in range(n)]))

def train(data):
    total_count = 0
    word_count = collections.defaultdict(lambda :0)
    bigram_count = collections.defaultdict(lambda :0)
    word_probabilities_1 = collections.defaultdict(lambda :0)
    word_probabilities_2 = collections.defaultdict(lambda :0)
    for line in data:
        line = '<s> ' + line
        line = line.lower().strip() + ' </s>'
        bigram = ngram(line.split(), 2)
        total_count += len(bigram) + 1
        for word_1, word_2 in bigram:
            bigram_count[word_1 + ' ' + word_2] += 1
            word_count[word_1] += 1
    for key in word_count.keys():
        word_probabilities_1[key] = word_count[key] / total_count
    for key in bigram_count.keys():
        word_probabilities_2[key] = bigram_count[key]/word_count[key.split()[0]]

    return word_probabilities_1, word_probabilities_2
if __name__ == "__main__":
    with open('../../data/wiki-en-train.word', 'r') as data:
        word_probabilities_1, word_probabilities_2 = train(data)
    for i, j in word_probabilities_2.items():
        print(i, j)
