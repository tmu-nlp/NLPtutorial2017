from collections import defaultdict, Counter
import pickle

train_f = open('../../data/wiki-en-test.norm').readlines()


def Create_Features(X, Y):
  phi = Counter()
  for i in range(len(Y)+1):
    if i == 0:
      first_tag = '<s>'
    else:
      first_tag = Y[i-1]
    if i == len(Y):
      next_tag = '</s>'
    else:
      next_tag = Y[i]
    phi.update(Create_Trans(first_tag, next_tag))
  for i in range(len(Y)):
    phi.update(Create_Emit(Y[i], X[i]))
  return phi

def Create_Trans(first_tag, next_tag):
  trans = defaultdict(int)
  trans[first_tag + ' ' + next_tag] += 1
  return trans

def Create_Emit(y, x):
  emit = defaultdict(int)
  emit[y + ' ' + x] += 1
  emit[y] += 1
  return emit

def Predict(w, phi):
  score = 0
  for name, value in phi.items():
    score += value * w[name]
  return score


def Hmm_Viterbi(w, words, trans, possible_tags):
  l = len(words)
  best_score = {}
  best_edge = {}
  best_edge['0 <s>'] = None
  best_score['0 <s>'] = 0
  for i in range(l):
    for prev in possible_tags:
      i_prev = '{} {}'.format(i, prev)
      if i_prev not in best_score: 
        continue
      for nexttag in possible_tags:
        prev_next = '{} {}'.format(prev,nexttag)
        if prev_next in trans:
          score = best_score[i_prev] + Predict(w, Create_Trans(prev, nexttag)) + Predict(w, Create_Emit(nexttag, words[i]))
          i1_next = '{} {}'.format(i+1, nexttag)
          if i1_next not in  best_score or  best_score[i1_next] < score:
            best_score[i1_next] = score
            best_edge[i1_next] = i_prev
  i = len(words)
  for prev in possible_tags:
    nexttag = '</s>' 
    i_prev = '{} {}'.format(i, prev)
    prev_next = '{} {}'.format(prev,nexttag)
    if i_prev  in best_score:
      score = best_score[i_prev] + Predict(w, Create_Trans(prev, nexttag)) + Predict(w, Create_Emit(nexttag, '</s>'))
      i1_next = '{} {}'.format(i + 1, nexttag)
      if i1_next not in  best_score or best_score[i1_next] < score:
        best_score[i1_next] = score
        best_edge[i1_next] = i_prev
  Y_hat = []
  next_edge = best_edge['{} {}'.format((l + 1),'</s>')]
  while next_edge != '0 <s>':
    position, tag = next_edge.split()
    Y_hat.append(tag)
    next_edge = best_edge[next_edge]
  Y_hat.reverse()
  return Y_hat  

def Update_Weights(w, phi_hat, phi_prime):
  keys = set(list(phi_hat.keys()) + list(phi_prime.keys()))
  #正解と予測の差を重みに加算
  for key in keys:
    w[key] += phi_prime[key] - phi_hat[key] 
  return w
  
  
if __name__ == '__main__':
  w = defaultdict(int)
  possible_tags = []
  trans = []
  size = len(train_f)
  w, possible_tags, trans = pickle.load(open('model_file.dump', 'rb'))
  for num, line in enumerate(train_f):
    line = line.strip().split()
    X = []
    Y_prime = []
    for word in line:
      X.append(word)
    import time
    s = time.time()
    Y_hat = Hmm_Viterbi(w, X, trans, possible_tags)
    # print(time.time() - s)
    print(' '.join(['{}_{}'.format(word, tag) for word, tag in zip(X, Y_hat)]))    

