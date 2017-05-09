from collections import defaultdict
counts = defaultdict(int)
total_count = 0

with open('../../data/wiki-en-train.word') as text:
    for line in text:
        words = line.split()
        words.append("</s>")
        for word in words:
            counts[word] += 1
            total_count += 1

with open('model_file.txt','a') as text:
    for y,w in counts.items():
        probability = float(w/total_count)
        text.write(y+"\t"+str(probability)+"\n")


    

