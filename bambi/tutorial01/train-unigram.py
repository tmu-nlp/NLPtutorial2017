from collections import defaultdict
counts = defaultdict(lambda: 0)
total_count = 0
file_path = "../../data/wiki-en-train.word"
model_file = "model.txt"
with open(file_path, "r") as file, open(model_file,"w") as output:
    for line in file.readlines():
        words = line.split()
        words.append("</s>")
        for word in words:
            counts[word] += 1
            total_count += 1
    for k,v in counts.items():
        probability = v/total_count
        print(k,probability)
        output.write("{} {}\n".format(k,probability))
