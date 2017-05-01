import sys
from collections import defaultdict

w_cnt = defaultdict(lambda: 0)

with open(sys.argv[1], 'r') as f:
    for line in f:
        line = line.lower()
        w_list = line.split()
        for i in range(len(w_list)):
            w_cnt[w_list[i]] +=1

for wd, ct in sorted(w_cnt.items()):
    print("{} {}" .format(wd, ct))
