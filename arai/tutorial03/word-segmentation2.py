import math
from collections import defaultdict

w_count = defaultdict(lambda :0)
t_count = 0
probabilities = dict()

with open('model_file.txt') as text:
    unigram = {}
    for i in text:
        i = i.strip().split()
        words = i
        words.append('<\s>')
        unigram[i[0]] = float(i[1])
        for word in words:
            w_count[word] += 1
            t_count += 1
    for word,count in w_count.items():
        probability = count/t_count
        probabilities[word] = probability
       
        

with open ('../../data/wiki-ja-test.txt') as text:
    lambda1 = 0.95
    V = 10**6

    for line in text:
        best_edge = {}
        best_score = {}
        best_edge[0] = None
        best_score[0] = 0
        for word_end in range(1,len(line)):
#            print(word_end)
            best_score[word_end] = 10**10
            for word_begin in range(word_end):
                #print(word_end)
                word = line[word_begin:word_end]
                #print(word_end)
                if (word in unigram) or (len(word)==1):

#                    prob = lambda1*float(unigram[word])+(1-lambda1)/V 
                    prob = probability 
                    my_score = best_score[word_begin] + math.log(prob)*-1
                    #print(word_end,my_score,best_score[word_end])
                    
                    if my_score<best_score[word_end]:
                        best_score[word_end] = my_score
                        best_edge[word_end] = (word_begin,word_end)
#                        print(word_begin,word_end)

        words = []
        next_edge = best_edge[len(best_edge)-1]
        while next_edge != None:
           word = line[next_edge[0]:next_edge[1]]
           word = line[next_edge[0]:next_edge[1]]
           words.append(word)
           next_edge = best_edge[next_edge[0]]
        words.reverse()
        print(" ".join(words))
        #print(words)
                
