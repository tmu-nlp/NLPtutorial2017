import sys
from collections import defaultdict

w_cnt = defaultdict(lambda: 0)
c_cnt = defaultdict(lambda: 0)

with open(sys.argv[1], 'r') as f:
    for line in f:
#        line = line.lower()
        w_list = line.split()
        w_list.append("</s>")
        w_list.insert(0,"<s>")
        #print(w_list)
        for i in range(1,len(w_list) ):
            #先ずは、key+ブランク+keyで、あとで、タプルで
            w_cnt[ w_list[i-1] + " " + w_list[i] ] += 1
            c_cnt[w_list[i-1]] += 1
            w_cnt[ w_list[i]] += 1
            c_cnt[""] += 1


#print(w_cnt.items())
#print(c_cnt.items())

with open('bigram_model.txt', 'w' ) as f2:
    for wd, ct in sorted(w_cnt.items()):
        wds = wd.split()
        if len(wds) ==2:
            #print(type(wds
            context = wds[0]
        else:
            context = ""
        #print(context)
        prob = float(w_cnt[wd])/c_cnt[context]

        print( "{}\t{}".format(wd, prob), file = f2)
