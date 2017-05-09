import math
with open('model_file.txt') as text:
    probabilities = {}
    for line in text:
        words = line.split()
        probabilities[words[0]] = float(words[1])

with open('../../data/wiki-en-test.word') as text:
   W=0
   unk=0
   H=0
   for line in text:
        words = line.split()
        words.append("</s>")
        for word in words:
            W+=1
            P = 0.05/1000000
            if word in probabilities:
                P+=0.95*probabilities[word]
            else:
                unk+=1
            H+=math.log(P)*-1
print("entropy =" +str(H/W))
print("coverage =" +str((W-unk)/W))




