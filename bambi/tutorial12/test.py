import math
import pickle
from train import *
from collections import defaultdict

w,possible_tags,transition = pickle.load(open("model.pickle","rb"))
file = "../../data/wiki-en-test.norm"
#file = "../test/05-test-input.txt"
with open("answer.pos","w") as output:
    for line in open(file):
        words = line.strip("\n").split(" ")
        Y_hat = HMM_VITERBI(w,words,transition,possible_tags)
        print(" ".join(Y_hat),file=output)
print("finished")
