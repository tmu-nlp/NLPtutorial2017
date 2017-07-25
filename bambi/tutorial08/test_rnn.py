import numpy as np
import pickle
from train_rnn import forward_rnn, create_ids, create_one_hot
from collections import defaultdict
#file = "../test/05-test-input.txt"
file = "../data/wiki-en-test.norm"
net = pickle.load(open( "net.pickle", "rb" ))
wRx,wRh,br,wOh,bo = net

x_ids = defaultdict(int)
for line in open("x_ids.txt"):
    name, value = line.split("\t")
    x_ids[name] = int(value)

y_ids = defaultdict(int)
for line in open("y_ids.txt"):
    name, value = line.split("\t")
    y_ids[name] = int(value)

keys = dict()
def id_to_pos(num):
    return keys[num]

for k,v in y_ids.items():
    keys[v] = k

with open("my_answer.pos", "w") as output:
    for line in open(file):
        x_list = []
        wordtags = line.strip("\n").split(" ")
        for wordtag in wordtags:
            if wordtag in x_ids:
                X = create_one_hot(x_ids[wordtag],len(x_ids))
                x_list.append(X)
            else:
                x_list.append(np.ones(len(x_ids))/len(x_ids))
        h,p,y_list = forward_rnn(wRx,wRh,br,wOh,bo,x_list)
        y_ = map(id_to_pos,y_list)
        print(" ".join(y_),file=output)
    print("finished")
