import sys
import math
from collections import defaultdict
word_count = defaultdict(int)
total_count = 0
probabilities = dict()

best_edge = dict()
best_score = dict()

lambda1 = .95
lambda_unk = 1 - lambda1
V = 10 ** 6
INF = 10 ** 10

with open('../../data/wiki-ja-train.word', 'r') as uni_file:
#with open('../../data/wiki-ja-test.word', 'r') as uni_file:
    for line in uni_file:
        words = line.split()
        words.append('</s>')
        for word in words:
            word_count[word] += 1
            total_count += 1
for word, count in word_count.items():
    probability = count/total_count
    probabilities[word] = probability
#print (probabilities+'\n')
count = 0
inner_couunt = 0
kitikiticount = 0
with open('../../data/wiki-ja-test.txt', 'r') as test_file, open('my_answer.word', 'w') as ans_file:
    for line in test_file:
        #前向きステップ
        best_edge = dict()
        best_score = dict()
        best_edge[0] = None
        best_score[0] = 0
        for word_end in range(1, len(line)+1):
            best_score[word_end] = INF
            for word_begin in range(0, word_end):
                word = line[word_begin:word_end]
                #print ('word :'+word)
                if word in probabilities or len(word) == 1:
                   P = lambda_unk / V
                   if word in probabilities:
                       P += lambda1 * probabilities[word]
                       #print ('knowing word : '+str(P))
                   """    
                   else:
                       print ('unknowing word:' +str(P))
                   """
                   my_score = best_score[word_begin] + -math.log2(P)
                   if my_score < best_score[word_end]:
                       best_score[word_end] = my_score
                       best_edge[word_end] = (word_begin, word_end)
                       # print ('!!!best_score_update!!!')
                       # inner_couunt += 1
                       # print ('inner cont' + str(inner_couunt))
                       """
                       if best_score[word_end] < best_score[word_end-1]:
                           print('おそらく既知語処理なんじゃないかなぁ')
                           kitikiticount += 1
                       """    
            #print('2for end word :'+word)           
        #print ('my_score : {},count : {}'.format(my_score, count))
        #count += 1
        #print (best_score)
        #print (best_edge)
        #print (kitikiticount)
        #print ('既知語が...')
        #後ろ向きステップ               
        words = []
        next_edge = best_edge[len(best_edge)-1]
        #print ('疑似コードママ')
        #print (len(best_edge)-1)
        #print (best_edge)
        #print ('line.length')
        #print (len(line))
        #next_edge = best_edge[len(line)]
        #print ('next_edge')
        #print (next_edge)
        while next_edge != None:
            word = line[next_edge[0]:next_edge[1]]
            words.append(word)
            next_edge = best_edge[next_edge[0]]
        words.reverse()
        ans_file.write(' '.join(words).strip(' '))
        #print (' '.join(words))
        #break
