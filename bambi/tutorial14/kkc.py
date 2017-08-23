from collections import defaultdict
import math

lm_file = "lm.txt"
tm_file = "tm.txt"
test_file = "../../data/wiki-ja-test.pron"
lm_probs = defaultdict(float)
for line in open(lm_file):
    words = line.strip("\n").split("\t")
    lm_probs[words[0]] = float(words[1])

tm_lm_probs = defaultdict(dict)
lamb = 0.0005
for line in open(tm_file):
    E, word, pron, prob = line.strip("\n").split(" ")
    tm_lm_probs[pron][word] = float(prob)

def key(*names):
    return " ".join(map(str,names))

def cal_score_front(prev_score,lm,tm):
    lm_p = lamb+(1-lamb)*lm # add lamb as smoothing
    inside_log = lm_p * tm
    if inside_log == 0:
        inside_log = 1
    return float(prev_score) - math.log(inside_log,2)

def cal_score_back(prev_score,prob):
    inside_log = lamb+(1-lamb)*prob
    if inside_log == 0:
        inside_log = 1
    return float(prev_score) - math.log(inside_log,2)

def kkc(line):
    line = line.strip("\n")
    edge = defaultdict(dict)
    score = defaultdict(dict)
    edge[0]["<s>"] = None
    score[0]["<s>"] = 0
    for end in range(1,len(line)+1):
        my_edges = defaultdict(int)
        for begin in range(end):
            pron = line[begin:end]
            my_tm = tm_lm_probs[pron]
            if len(my_tm.keys()) == 0 and len(pron) == 1:
                my_tm = {pron:0}
            for curr_word, tm_prob in my_tm.items():
                for prev_word, prev_score in score[begin].items():
                    curr_score = cal_score_front(prev_score,lm_probs[key(prev_word,curr_word)],tm_prob)
                    if curr_word in score[end]:
                        # update only if
                        if curr_score < score[end][curr_word]:
                            score[end][curr_word] = curr_score
                            edge[end][curr_word] = {begin: prev_word}
                    else:
                        # add new key
                        score[end][curr_word] = curr_score
                        edge[end][curr_word] = {begin: prev_word}
    for prev, prev_score in score[end].items():
        s = cal_score_back(prev_score,lm_probs[key(prev,'</s>')])
        if "</s>" in score[end+1]:
            if score[end+1]["</s>"] > s:
                score[end+1]["</s>"] = s
                edge[end+1]["</s>"] = {end:prev}
        else:
            score[end+1]["</s>"] = s
            edge[end+1]["</s>"] = {end:prev}

    tags = []
    next_edge = edge[end+1]["</s>"]
    while 0 not in next_edge:
        position, tag = next_edge.popitem()
        tags.append(tag)
        next_edge = edge[position][tag]
    tags.reverse()
    answer = "".join(tags)
    return answer

with open("output.txt","w") as output:
    for line in open(test_file):
        content = kkc(line)
        print(content,file=output)
print("finished")
