from collections import defaultdict

cnt = defaultdict(lambda :0)

with open('../../data/wiki-en-train.word') as f:
    for line in f:
        l = line.strip().replace(',', '').replace('.','').split(' ')
        a = [w.lower() for w in l if len(w) > 0]
        for w in a :
            cnt[w] += 1

for k,v in sorted(cnt.items(), key=lambda x:x[1], reverse=True):
    print(k, v)

print(len(cnt))