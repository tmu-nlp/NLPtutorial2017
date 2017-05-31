from collections import defaultdict
import math


V = 10**6
lambda1 = 0.95
with open('model.txt') as model_file:
    transition = defaultdict(float)
    emission = defaultdict(float)
    possible_tags = dict()
    for line in model_file:
        tag, context, word, prob = line.strip().split()
        possible_tags[context] = 1
        if tag == "T":
            transition[context +" "+ word] = float(prob)
        else:
            emission[context +" "+ word] = float(prob)


with open('../../data/wiki-en-test.norm') as text:
    for line in text:
        words = line.strip().split()
        l = len(words)
        best_score = dict()
        best_edge = dict()
        best_score["0 <s>"] = 0
        best_edge["0 <s>"] = 'NULL'
        for i in range(l):
            for prev in possible_tags.keys():
                for next_word in possible_tags.keys():
                    if (str(i) + " " + prev) in best_score and (prev + " " + next_word) in transition:
                        score = float(best_score[str(i) + " " + prev]) + float(math.log2(transition[prev + " " + next_word])) * (-1.0) + float(math.log2(lambda1 * emission[next_word + " " + words[i]] + (1.0 - lambda1) / V) * (-1.0))
                        if (str(i + 1) + " " + next_word) not in best_score or best_score[str(i + 1) + " " +  next_word] > score:
                            best_score[str(i + 1) + " " + next_word] = score
                            best_edge[str(i + 1) + " " + next_word] = str(i) + " " + prev

        for prev in possible_tags.keys():
            if (str(len(words)) + " " + prev) in best_score and (prev + " " + "</s>") in transition:
                score = float(best_score[str(len(words)) + " " + prev]) + float(math.log2(transition[prev + " " + "</s>"])) * (-1.0) + float(math.log2(emission[prev + " " + "</s>"] + (1.0 - lambda1) / V) * (-1.0))
                if (str(len(words) + 1) + " " + "</s>") not in best_score or best_score[str(len(words) + 1) + " " + "</s>"] > score:
                    best_score[str(len(words) + 1) + " " + "</s>"] = score
                    best_edge[str(len(words) + 1) + " " + "</s>"] = str(len(words)) + " " + prev

    #    print(best_edge)                    
     
        tags = []
        next_edge = best_edge[str(l + 1)  + " " + "</s>"]
#        for key, value in best_edge.items():
 #           if value == 0:
  #              print('{}\t{}'.format(key, value))
        while next_edge != "0 <s>":
   #         print(next_edge)

            position, tag = next_edge.split()
            tags.append(tag)
            next_edge = best_edge[next_edge]
        tags.reverse()
        print(" ".join(tags))    

