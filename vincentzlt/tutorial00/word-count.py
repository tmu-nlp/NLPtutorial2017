#!/usr/bin/python
# -*- coding: UTF-8 -*-

# python word-count.py <filename>

import sys
from collections import defaultdict


def word_f(f_name):
    try:
        with open(f_name, "r",encoding="UTF-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("there is no such file.")
        return

    w_freq_dict = defaultdict(lambda: 0)
    for i in lines:
        for j in i.split():
            w_freq_dict[j]+=1
    return w_freq_dict

if __name__=="__main__":
    w_freq_dict=word_f(sys.argv[1])
    for i,j in w_freq_dict.items():
        print(i, "\t", str(j))
    print("# of types:\t", len(w_freq_dict))
