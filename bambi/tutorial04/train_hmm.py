from collections import defaultdict
emit = defaultdict(int)
transition = defaultdict(int)
context = defaultdict(int)
file_name = "../../data/wiki-en-train.norm_pos"
#file_name = "../test/05-train-input.txt"
for line in open(file_name):
    line = line.strip("\n")# clean line
    previous = "<s>"
    context[previous] += 1
    wordtags = line.split(" ")
    for wordtag in wordtags:
        word, tag = wordtag.split("_")
        transition["{} {}".format(previous,tag)] += 1
        context[tag] += 1
        emit["{} {}".format(tag,word)] += 1
        previous = tag
    transition["{} </s>".format(previous)] += 1

result = ""

# Print the transition probabilities
for key, value in sorted(transition.items()):
    previous, word = key.split(" ")
    p_trans = "T {} {}\n".format(key, value/context[previous])
    result += p_trans

# Print the transition probabilities
for key, value in sorted(emit.items()):
    tag, word = key.split(" ")
    p_emit = "E {} {}\n".format(key, value/context[tag])
    result += p_emit

with open("model_hmm.txt","w") as output:
    print(result)
    output.write(result)
