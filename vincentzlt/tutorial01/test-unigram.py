#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import math
from collections import defaultdict

def read_model(f_name):
    """
    read from file
    return a dict
        key:    word
        value:  P"""
    unigram_model=dict()
    with open(f_name,"r") as f:
        lines=f.readlines()
    for i in [j.split() for j in lines]:
        try:
            unigram_model[i[0]]=float(i[1])
        except IndexError:
            print(i)
            input()
    return unigram_model

def calc_entropy(f_name,model):
    """
    input:
        f_name: corpus file to be calc
        model:  unigram model as dict read from function read_model
    output:
        entropy, coverage
        """
    with open(f_name,"r",encoding="UTF-8") as d:
        lines=d.readlines()
    lines=[(i+" </s>").split() for i in lines]

    h=0
    num_token=0
    num_unk_word=0
    for i in lines:
        for j in i:
            if j in model:
                h += -math.log2(model[j] * 0.95 + 1 / float(1000000) * 0.05)
            else:
                h+= -math.log2(1/float(1000000)*0.05)
                num_unk_word+=1
            num_token+=1
    h/=num_token
    coverage=1-num_unk_word/float(num_token)
    return h,coverage



if __name__=="__main__":
    model=read_model(sys.argv[1])
    print(calc_entropy(sys.argv[2],model))
