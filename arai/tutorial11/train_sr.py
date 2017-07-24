from collections import defaultdict
import pickle
import sys


def PredictScore(weight, features):
  score = 0
  for name, value in features.items():
    if name in weight:
      score += value * weight[name]
  return score

def UpdateWeights(weight, features, predict, correct):
  names = ['shift', 'left', 'right']
  for name, value in features.items():
    weight[names.index(correct)][name] += value   
    weight[names.index(predict)][name] -= value

def ShiftReduceTrain(queue, heads, weights):
  stack = [(0, 'ROOT', 'ROOT')]
  weight_shift, weight_left, weight_right = weights
  unproc = list()
  for i in range(len(heads)):
    unproc.append(heads.count(i))
  while len(queue) > 0 or len(stack) > 1:

    features = MakeFeatures(stack, queue)
    score_shift = PredictScore(weight_shift, features)
    score_left = PredictScore(weight_left, features)
    score_right = PredictScore(weight_right, features)
 
    if (score_shift >= score_left and score_shift >= score_right and len(queue) > 0) \
        or (len(stack) < 2):
      predict = 'shift'
    elif score_left >= score_right :
      predict = 'left'
    else:
      predict = 'right'

    if len(stack) < 2:
      correct = 'shift'
    elif heads[stack[-1][0]] == stack[-2][0] and unproc[stack[-1][0]] == 0:
      correct = 'right'
    elif heads[stack[-2][0]] == stack[-1][0] and unproc[stack[-2][0]] == 0:
      correct = 'left'
    else:
      correct = 'shift'

    if predict != correct:
      UpdateWeights(weights, features, predict, correct)
    if correct == 'shift':
      if queue != []:
        stack.append(queue.pop(0))
    elif correct == 'left':
      unproc[stack[-1][0]] -= 1
      stack.pop(-2)
    elif correct == 'right':
      unproc[stack[-2][0]] -= 1
      stack.pop(-1)

def MakeFeatures(stack, queue):
  features = defaultdict(int)
  if len(stack) > 0 and len(queue) > 0:
    features['W-1' + stack[-1][1] + ',WO' + queue[0][1]] += 1
    features['W-1' + stack[-1][1] + ',PO' + queue[0][2]] += 1
    features['P-1' + stack[-1][2] + ',WO' + queue[0][1]] += 1
    features['P-1' + stack[-1][2] + ',PO' + queue[0][2]] += 1
  if len(stack) > 1:
    features['W-2' + stack[-2][1] + ',W-1' + stack[-1][1]] += 1
    features['W-2' + stack[-2][1] + ',P-1' + stack[-1][2]] += 1
    features['P-2' + stack[-2][2] + ',W-1' + stack[-1][1]] += 1
    features['P-2' + stack[-2][2] + ',P-1' + stack[-1][2]] += 1
  return features


      

if __name__ == '__main__':
  r_file = open('../../data/mstparser-en-train.dep').readlines()
  w_file = open('train_file.pkl', 'wb')
  data = list()
  queue = list()
  heads = [-1]
  for line in r_file:
    if line.strip() == '':
      data.append([queue, heads])
      queue = list()
      heads = [-1]
    else:
      ID, word, base, pos, pos2, _, head, label = line.strip().split()
      queue.append([int(ID), word, pos])
      heads.append(int(head))

  weight_shift = defaultdict(float)
  weight_left = defaultdict(float)
  weight_right = defaultdict(float)
  l = sys.argv[1]
  weights = [weight_shift, weight_left, weight_right]

  size = len(data)
  for i in range(int(l)):
    for num, (queue, heads) in enumerate(data):
      ShiftReduceTrain(queue, heads, weights)
      print("{:2f}%".format(100* num / size), end="\r")
    print(i)
  pickle.dump(weights, w_file)
    

