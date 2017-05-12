import collections

def train(data):
    total_count = 0
    word_count = collections.defaultdict(lambda :0)
    word_probabilities = collections.defaultdict(lambda :0)
    for line in data:
        for word in line.split():
            if word == '.':
                continue
            total_count += 1
            word_count[word.lower()] += 1
    for word in word_count.keys():
        word_probabilities[word] = word_count[word]/total_count
    return word_probabilities

if __name__ == "__main__":
    with open('../../data/wiki-en-train.word', 'r') as data:
        for i, j in train(data).items():
            print(i, j)
