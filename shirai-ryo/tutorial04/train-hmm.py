from collections import defaultdict
emit = defaultdict(int)
transition = defaultdict(int)
context = defaultdict(int)

with open("../../data/wiki-en-train.norm_pos") as text:
    #テストファイルなら ../../test/05-train-input.txt
    #テストファイルで実行中。あとで戻すのを忘れずに。（もどした）
    #擬似コードの「＋＋」が「＋＝１」なのかよくわかってないのであとで確認する
    #あってた
    for line in text:
        previous = "<s>"
        context[previous] += 1
        wordtags = line.strip().split(" ")
        for wordtag in wordtags:
            word, tag = wordtag.split("_")
            #splitされたものがwordとtagにそれぞれ入る
            #natural_JJがword = naturalとtag = JJみたいな感じで
            transition[previous + " " + tag] += 1 #遷移
            context[tag] += 1 #文脈
            emit[tag + " " + word] += 1 #生成
            previous = tag
            #前の単語の種類が何だったかを更新してるっぽい
        transition[previous + " </s>"] += 1

# print(emit)
# print(transition)
# print(context)


with open("model_file.word", "w") as text:
    for key, value in transition.items():
        previous, word = key.split()
        value = value/context[previous]
        text.write("T" + " " + key + " " + str(value) + "\n")
    for key, value in emit.items():
        previous, word = key.split()
        value = value/context[tag]
        text.write("E" + " " + key + " " + str(value) + "\n")

#model_fileがなんかずれるので質問する
