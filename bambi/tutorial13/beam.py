from collections import defaultdict
import math

file = "../../data/wiki-en-test.norm"

emission = defaultdict(int)
transition = defaultdict(lambda: 1) # 1 to prevent log(0) raise math domain error and the log(1) == 0
possible_tags = defaultdict(int)

def key(*names):
    return " ".join(map(str,names))

def cal_score(best_score,transition,score_key,trans_key,emis_key):
    '''
    PT(yi|yi-1) = PML(yi|yi-1)
    PE(xi|yi) = λ PML(xi|yi) + (1-λ) 1/N
    '''
    lamb_ = 0.95
    V = 1000000
    P_emit = math.log(lamb_ * emission[emis_key] + ((1.0-lamb_)/V),2)
    return float(best_score[score_key]) - float(math.log(transition[trans_key],2))  - float(P_emit)

for line in open("model_hmm.txt"):
    word_type, context, word, prob = line.strip("\n").split(" ")
    possible_tags[context] = 1
    # enumerate all tags
    if word_type == "T":
        transition["{} {}".format(context,word)] = float(prob)
    else:
        emission["{} {}".format(context,word)] = float(prob)


B = 10
def beam(line):
    words = line.strip("\n").split()
    l = len(words)
    best_score = {}
    best_edge = {}
    active_tags = []
    active_tags.append(["<s>"])
    best_score["0 <s>"] = 0 # Start with <s>
    best_edge["0 <s>"] = None
    for i in range(0, len(words)):
        my_best = {}#add from previous hmm
        for prev in active_tags[i]: #add from previous hmm
            for next_tag in possible_tags.keys():
                if key(i,prev) in best_score and key(prev,next_tag) in transition:
                    score = cal_score(best_score,transition,key(i,prev),key(prev,next_tag),key(next_tag, words[i]))
                    if key(i+1,next_tag) not in best_score or best_score[key(i+1,next_tag)] > score:
                        best_score[key(i+1,next_tag)] = score
                        best_edge[key(i+1,next_tag)] = key(i,prev)
                        my_best[next_tag] = score #add from previous hmm

        my_best_B_elements = [k for k, v in sorted(my_best.items(), key = lambda x:x[1])[:B]]# small v is better
        active_tags.append(my_best_B_elements)


    for prev in active_tags[i]:#add from previous hmm
        score_key = "{} {}".format(l, prev)
        trans_key = "{} </s>".format(prev)
        emis_key = trans_key
        last_score_key = "{} </s>".format(l+1)
        if score_key in best_score:
            score = cal_score(best_score,transition,score_key,trans_key,emis_key)
            if last_score_key not in best_score or best_score[last_score_key] > score:
                best_score[last_score_key] = score
                best_edge[last_score_key] = score_key
    tags = []
    next_edge = best_edge["{} </s>".format(l+1)]
    while next_edge != "0 <s>":
        position, tag = next_edge.split()
        tags.append(tag)
        next_edge = best_edge[next_edge]
    tags.reverse()
    answer = " ".join(tags)
    print(answer)
    return answer

with open("beam.pos","w") as output:
    for line in open(file):
        print(beam(line), file=output)

print("finished")
