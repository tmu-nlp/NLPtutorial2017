from collections import defaultdict
import math
emission = defaultdict(int)
transition = defaultdict(int)
possible_tags = defaultdict(int)

file_name = "../../data/wiki-en-test.norm"

#file_name = "../test/05-test-input.txt"

for line in open("model_hmm.txt"):
    word_type, context, word, prob = line.strip("\n").split(" ")
    possible_tags[context] = 1
    # enumerate all tags
    if word_type == "T":
        transition["{} {}".format(context,word)] = float(prob)
    else:
        emission["{} {}".format(context,word)] = float(prob)

def cal_score(best_score,transition,score_key,trans_key,emis_key):
    '''
    PT(yi|yi-1) = PML(yi|yi-1)
    PE(xi|yi) = λ PML(xi|yi) + (1-λ) 1/N
    '''
    lamb_ = 0.95
    V = 1000000
    P_emit = math.log(lamb_ * emission[emis_key] + (1-lamb_)/V,2)
    return best_score[score_key] - math.log(transition[trans_key],2) - P_emit


def viterbi(line):
    line = line.strip("\n")
    words = line.split()
    l = len(words)
    best_score = {}
    best_edge = {}
    best_score["0 <s>"] = 0 # Start with <s>
    best_edge["0 <s>"] = None

    for i in range(0, len(words)):
        for prev in possible_tags.keys():
            for next_tag in possible_tags.keys():
                score_key = "{} {}".format(i,prev)
                trans_key = "{} {}".format(prev,next_tag)
                emis_key = "{} {}".format(next_tag, words[i])
                if score_key in best_score and trans_key in transition:
                    score = cal_score(best_score,transition,score_key,trans_key,emis_key)
                    next_score_key = "{} {}".format(i+1,next_tag)
                    if next_score_key not in best_score or best_score[next_score_key] > score:
                        best_score[next_score_key] = score
                        best_edge[next_score_key] = score_key

    for prev in possible_tags.keys():
        score_key = "{} {}".format(l, prev)
        trans_key = "{} </s>".format(prev)
        next_score_key = "{} </s>".format(l+1)
        if score_key in best_score and trans_key in transition:
            score = best_score[score_key] - math.log(transition[trans_key], 2)
            if next_score_key not in best_score or best_score[next_score_key] > score:
                best_score[next_score_key] = score
                best_edge[next_score_key] = score_key = score_key
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

result = ""
for line in open(file_name, "r"):
    result += viterbi(line)
    result += "\n"
with open("my_answer.pos","w") as output:
    output.write(result)
