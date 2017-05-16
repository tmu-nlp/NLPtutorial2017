from collections import defaultdict

t_f = open('wiki-en-train.word','r')

w_count = defaultdict(lambda: 0)
t_count = 0

for line in t_f:
    words = line.split()
    words.append('<\s>')
    for word in words:
        w_count[word.lower()] += 1
        t_count += 1

m_f = open('train-model.txt','w')
for word,count in (w_count.items()):
    probability = count/t_count
    model = ('{} {}'.format(word,probability))
    m_f.write(model + '\n')
