import sys
from collections import defaultdict

f = open(sys.argv[1], "r")

counts = defaultdict(lambda: 0)
for line in f:
    line = line.replace(',','')
    line = line.replace('.','')
    line = line.split()
    l = len(line)
    for i in range(l):
        counts[line[i]] += 1

for k, v in sorted(counts.items(), key = lambda x:x[1], reverse = True):
    print("%s : %d" %(k,v))
