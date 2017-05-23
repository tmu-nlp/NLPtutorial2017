import math
from collections import defaultdict
counts = defaultdict(int)
total_count = 0

with open("../../data/wiki-ja-train.word") as text:
    for line in text:
        words = line.split()
        words.append("</s>")
        #ダメだったら分ける
        for word in words:
            counts[word] += 1
            total_count += 1

with open("model_file.word", "w") as text:
    my_dict = {}
    for word, count in counts.items():
        #.keys()はキー
        #.values()はバリュー
        probability = float(counts[word]/total_count)
        text.write(word + "\t" + str(probability) + "\n")
        my_dict[word] = probability


#ここまで1-gramのやつ

#前向きステップ

with open("../../data/wiki-ja-test.txt") as text:
    a = 0.05 #ラムダ１のこと
    V = 1000000

    for line in text:
        line = line.strip()
        best_edge = {}
        best_score = {}
        best_edge[0] = "NULL"
        best_score[0] = 0
        for word_end in range(1,len(line) + 1):
            best_score[word_end] = 10 ** 10
            for word_begin in range(word_end):
                word = line[word_begin:word_end]
                if word in my_dict or len(word) == 1:
                    prob = a * probability + (1 - a) / V
                    my_score = best_score[word_begin] + math.log(probability, 2) * -1
                    if my_score < best_score[word_end]:
                        best_score[word_end] = my_score
                        best_edge[word_end] = (word_begin, word_end)
        # print(line)
        # print(best_edge)
        # break

#後ろ向きステップ

        words = []
        next_edge = best_edge[len(best_edge) - 1]
        while next_edge != "NULL":
            word = line[next_edge[0]: next_edge[1]]
            words.append(word)
            next_edge = best_edge[next_edge[0]]
        words.reverse()
        words = " ".join(words)
        print(words)


        with open("my_answer.word", "a") as answer:
            answer.write(words + "\n")
