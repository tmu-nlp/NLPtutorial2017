from collections import defaultdict
import math
unigram_probs = defaultdict(int)
# load unigram-probs model
#unigram_model_path = "../test/04-model.txt"
unigram_model_path = "model_jp.txt"
for line in open(unigram_model_path, "r"):
    words = line.split()
    unigram_probs[words[0]] = float(words[1])
#input_path = "../test/04-input.txt"
input_path = "../../data/wiki-ja-test.word"

# for exercise 1
lambda_1 = 0.95
unknown_lambda = 1 - lambda_1
volume = 1000000.0
P = unknown_lambda/volume

answer = ""
for line in open(input_path, "r"):
    best_edge = {}
    best_score = {}
    #forward step
    line = line.strip().replace(" ", "") # python3, str is unicode by default so no need to encode
    best_edge[0] = None
    best_score[0] = 0
    '''
    range(1,5) => [1,2,3,4] ; start <= x < end,
    so the instruction said [1,0,..length(line)] => range(1, len(line) +1); +1 to make len(line) be member of []
    '''
    for word_end in range(1,len(line)+1):
        best_score[word_end] = 10 ** 10
        for word_begin in range(0,word_end):
            word = line[word_begin:word_end] # get the substring
            if word in unigram_probs or len(word) == 1: # only known words
                prob = P + lambda_1 * unigram_probs[word]
                my_score = best_score[word_begin] - math.log(prob,2)
                if my_score < best_score[word_end]: # best score means shorter one
                    best_score[word_end] = my_score
                    best_edge[word_end] = (word_begin, word_end)
    #backward step
    words = []
    key = len(best_edge) - 1
    next_edge = best_edge[key]
    while next_edge != None:
        # add the substring for this edge to words
        word = line[next_edge[0]:next_edge[1]]
        words.append(word)
        next_edge = best_edge[next_edge[0]]
    words.reverse()
    answer = " ".join(words)
    print(answer)

with open("my_answer.word","w") as output:
    output.write(answer)
