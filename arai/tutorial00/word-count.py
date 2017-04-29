f=open('wiki-en-train.word','r')
word=f.read().replace(",","").replace(".","")
words=word.split()
word_and_counts={}
for i in words:
    if i in word_and_counts:
        word_and_counts[i] +=1
    else:
            word_and_counts[i] =1
for y,w in sorted(word_and_counts.items(),key=lambda x: x[1], reverse=True):
    print(y,w)
print(len(word_and_counts))
