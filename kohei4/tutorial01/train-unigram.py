import sys
from collections import defaultdict

w_cnt = defaultdict(lambda: 0)
t_cnt = 0

with open(sys.argv[1], 'r') as f:
    for line in f:
#        line = line.lower()
        w_list = line.split()
        w_list.append("</s>")
        for w in w_list:
            w_cnt[w] +=1
            t_cnt += 1

with open('model_file.txt', 'w' ) as f2:
    for wd, ct in sorted(w_cnt.items()):
        print("{}\t{}" .format(wd, float(ct)/t_cnt),file=f2)
