from collections import defaultdict
import math


V = 10**6
lambda1 = 0.95
with open('tm.txt') as model_file:
    transition = defaultdict(list)
    emission = defaultdict(list)
    possible_tags = dict()
    for line in model_file:
        tag, context, word, prob = line.strip().split()
        possible_tags[context] = 1
        if tag == "E":
            transition[word].append( [context, float(prob)])
        else:
            pass
            # transition[context +" "+ word] = float(prob)
    emission = dict(emission)
#    print(emission)
#exit()
lm_file = open('lm.txt').readlines()
emission = defaultdict(lambda: .000001)
for line in lm_file:
  try:
    context, word, prob = line.strip().split()
    emission[context +" "+ word] = float(prob)
  except:
    pass

with open('../../data/wiki-ja-test.pron') as text:
  for line in text:
    line = line.strip()
    l = len(line)
    best_score = defaultdict(lambda :defaultdict(float))
    best_edge = defaultdict(dict)
    best_score[0]["<s>"] = 0
    best_edge[0]["<s>"] = 'NULL'
    for end in range(1,len(line) + 1):
      for begin in range(end):
        pron = line[begin:end]
        my_tm = transition.get(pron, [])
        if my_tm == [] and len(pron) == 1:
          my_tm.append([pron,.000001])
        for curr_word, tm_prob in my_tm:
          for prev_word, prev_score in best_score[begin].items():
            curr_score = prev_score - math.log(tm_prob * emission[prev_word +" "+ curr_word])

            if curr_word not in best_score[end] or curr_score < best_score[end][curr_word]:
              best_score[end][curr_word] = curr_score
              best_edge[end][curr_word] = (begin, prev_word)
    # process eos
    curr_word = "</s>"
    end = len(line) + 1
    begin = len(line)
    for prev_word, prev_score in best_score[begin].items():
      curr_score = prev_score  -math.log(tm_prob * emission[pron +" "+ word])
      if curr_word not in best_score[end] or curr_score < best_score[end][curr_word]:
        best_score[end][curr_word] = curr_score
        best_edge[end][curr_word] = (begin, prev_word)

    # back_ward
    tags = []
    next_edge = best_edge[end][curr_word]
    while next_edge != (0, "<s>"):

        position, tag = next_edge
        tags.append(tag)
        next_edge = best_edge[position][tag]
    tags.reverse()
    print(" ".join(tags))    

