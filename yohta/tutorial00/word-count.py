import sys
f = open('wiki-en-train.word','r')

word_counts = {}
for l in f:
    l = l.strip()
    if len(l) != 0:
        words = l.split(" ")
        for word in words:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1

for word,n in sorted(word_counts.items(),key = lambda x:x[1],reverse = True):
    print(word + " " + str(word_counts[word]))
