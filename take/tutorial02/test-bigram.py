from collections import defaultdict
from math import log2

counts = defaultdict(int)
context_count = defaultdict(int)
u_counter = defaultdict(int)
c_counter = defaultdict(int)

#TRAINING_FILE = "sample.txt"
#TRAINING_FILE = "../../test/02-train-input.txt"
# TRAINING_FILE = "../../data/wiki-en-train.word"
TEST_FILE = "../../data/wiki-en-test.word"

GEN_MODEL_FILE = "02.model"

lambda1 = 0.95
lambda2 = 0.95

V=1000000
H=0
W=0

probs = defaultdict(int)

# def calc_lambda2(word):
#     return 1. - (u_counter(word)/(


with open(GEN_MODEL_FILE) as f:
    for l in f:
        t_list = l.strip('\n').split('\t')
        probs[t_list[0]] = float(t_list[1])
        # WittenBwll smoothingのために、bigramから異なり数を数える
        if ' ' in t_list[0]: # スペースを含むことから、それがbigramであることを判定
            print("contain space {} | {}".format(t_list[0], t_list[0].split(' ')[0]))
            # print(t_list[0].split(' ')[0])
            u_counter[t_list[0].split(' ')[0]] += 1 #異なり数

# print(u_counter)
# print(probs)

with open(TEST_FILE) as f:
    for line in f:
        l = line.strip('\n').lower().split(' ')
        l.insert(0,'<s>'), l.append('</s>')
        for i in range(1,len(l)):
            p1 = lambda1 * probs[l[i]] + (1. - lambda1) / V
            p2 = lambda2 * probs[" ".join(l[i-1:i+1])] + (1. - lambda2)*p1
            H += -log2(p2)
            W += 1
            # print("".join(l[i-1:i+1]))
            # counts[" ".join(l[i-1:i+1])] += 1
            # context_count[l[i-1]] += 1
            # counts[l[i]] += 1
            # context_count[''] += 1
print('entropy: ',H/W)
# with open(GEN_MODEL_FILE, mode="w") as f:
#     for ngram, cnt in counts.items():
#         temp_list = []
#         temp_list = ngram.split(' ')
#         temp_list[len(temp_list)-1] = ""
#         context_word = temp_list[0]
#         prob = counts[ngram]/context_count[context_word]
#         f.write(ngram + '\t' + str(prob) + '\n')
#         print('{0}\t{1}'.format(ngram, prob))
#         # print('ngram_ctx:{0}, ctxword:{1}, ngram:{2}, ctxwd:{3}'.format(counts[ngram], context_count[context_word], ngram, context_word))
