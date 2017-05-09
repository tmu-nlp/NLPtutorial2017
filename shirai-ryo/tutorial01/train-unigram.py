from collections import defaultdict
counts = defaultdict(int)
total_count = 0

with open("../../data/wiki-en-train.word") as text:
    for line in text:
        words = line.split()
        words.append("</s>")
        #ダメだったら分ける
        print(words)
        for word in words:
            counts[word] += 1
            total_count += 1

with open("model_file.word", "a") as text:
    for word, count in counts.items():
        #.keys()はキー
        #.values()はバリュー
        probability = float(counts[word]/total_count)
        text.write(word + "\t" + str(probability) + "\n")
