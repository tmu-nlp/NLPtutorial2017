from collections import defaultdict
import math

transition = defaultdict(int)
emission = defaultdict(int)
possible_tags = defaultdict(float)

a = 0.95
V = 10 ** 6

with open("model_file.word") as text:
    for line in text:
        tag, context, word, prob = line.split()
        #tagはチュートリアル資料ではtypeだけど、関数名にもあるから使えないっぽいのでtagにした
        possible_tags[context] = 1
        prob = float(prob)
        if tag == "T":
            transition[context + " " + word] = prob
        else:
            emission[context + " " + word] = prob
#多分ここまではおk

#前向きステップ
with open("../../data/wiki-en-test.norm") as text:
    possible_tags["</s>"] = 1
    for line in text:
        words = line.split()
        words.append("</s>")
        l = len(words)
        best_score = {}
        best_edge = {}
        best_score["0 <s>"] = 0
        best_edge["0 <s>"] = "NULL"
        active_tags[0] = ["<s>"]
        for i in range(l):
            my_list = dict()
            for prev in active_tags.keys():
            #最初にくる品詞をまわす
                for next_tag in possible_tags.keys():
                #その次にくる品詞をまわす
                #nextは関数にもあるので？、名前として使えない（typeと同様）
                    if str(i) + " " + prev in best_score and prev + " " + next_tag in transition:
                        # print(transition[prev + " " + next_tag])
                        # print(emission[next_tag + " " + words[i]])
                        score = best_score[str(i) + " " + prev] -math.log2(float(transition[prev + " " + next_tag])) -math.log2(float(a * emission[next_tag + " " + words[i]] + (1-a)/V))
                        if (str(i+1) + " " + next_tag not in best_score) or (best_score[str(i+1) + " " + next_tag] > score):
                            best_score[str(i+1) + " " + next_tag] = score
                            best_edge[str(i+1) + " " + next_tag] = str(i) + " " + prev
                            my_best[next] = score
                            #print(my_best)
            for key, value in sorted(my_best.items(), key=lambda x:x[1]):
                active_tags[i+1].append(key)
                if len(active_tags[i+1]) == B:
                    break


        #後ろ向きステップ
        tags = []
        next_edge = best_edge[str(l) + " </s>"]
        while next_edge != "0 <s>":
            position, tag = next_edge.split()
            tags.append(tag)
            next_edge = best_edge[next_edge]
        tags.reverse()
        tags = " ".join(tags)
        print(tags)

        with open("my_answer.word", "a") as answer:
             answer.write(tags)
             answer.write("\n")
