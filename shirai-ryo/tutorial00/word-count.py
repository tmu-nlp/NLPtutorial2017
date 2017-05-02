
f = open('wiki-en-train.word')
words = f.read().split()

print(words)


my_dict = {}

for w in words:
    if w in my_dict:
        my_dict[w] += 1
    else:
        my_dict[w] = 1

#異なり数
print(len(my_dict))
#頻度
for foo, bar in sorted(my_dict.items())[:10]:
    #print("%s: %d" % (foo,bar))
    print("{}: {}".format(foo,bar))
