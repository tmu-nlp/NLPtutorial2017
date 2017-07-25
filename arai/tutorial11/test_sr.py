from collections import defaultdict
import pickle
import train_sr

r_file = open('../../data/mstparser-en-test.dep').readlines()
w_file = open('train_file.pkl', 'rb')
data = list()

def PredictScore(weight, features):
  score = 0
  for name, value in features.items():
    if name in weight:
      score += value * weight[name]
  return score


def ShiftReduceTest(queue, weights):
  stack = [(0, 'ROOT', 'ROOT')]
  heads = [-1]*(len(queue) + 1)
  weight_shift, weight_left, weight_right = weights
  unproc = list()
  while len(queue) > 0 or len(stack) > 1:
    features = train_sr.MakeFeatures(stack, queue)
    score_shift = PredictScore(weight_shift, features)
    score_left = PredictScore(weight_left, features)
    score_right = PredictScore(weight_right, features)
 
    if (score_shift >= score_left and score_shift >= score_right and len(queue) > 0) \
        or (len(stack) < 2):
      if queue != []:
        stack.append(queue.pop(0))
    elif score_left >= score_right:
      heads[stack[-2][0]] = stack[-1][0]
      stack.pop(-2)
    else: 
      heads[stack[-1][0]] = stack[-2][0]
      stack.pop(-1)
  return heads



      

if __name__ == '__main__':
  data = list()
  queue = list()
  for line in r_file:
    if line.strip() == '':
      data.append(queue)
      queue = list()
    else:
      ID, word, base, pos, pos2, _, head, label = line.strip().split()
      queue.append([int(ID), word, pos])
  weights = pickle.load(w_file)
  for queue in data:
    x = list('a'*8)
    x[6] = '{}'
    x = '\t'.join(x)
    heads = ShiftReduceTest(queue, weights)
    print("\n".join([x.format(head)for head in heads[1:]]), end = '\n\n')
    

