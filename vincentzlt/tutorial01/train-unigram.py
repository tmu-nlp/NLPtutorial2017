#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import math
from collections import defaultdict


def preprocess_corpus(f_name):
    """
    input:  a file name
    ouput:  a list of lines in the corpus
            each line is a list"""

    try:
        with open(f_name, "r", encoding="UTF-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("file not found. try again")
    except:
        print(sys.exc_info())

    output = [(i + " </s>").split() for i in lines]

    return output


def calc_w_uni_P(w, corpus_list, *g_and_vocab_number):
    """
    input:
    w               the word
    corpus_list     the corpus that train the model
    g               the lambda value to unk
    vocab_number    the vocab number to construct the mnodel"""

    num_w = 0
    num_token = 0
    for i in corpus_list:
        for j in i:
            if j == w:
                num_w += 1
            num_token += 1
    if g_and_vocab_number:
        w_P = g_and_vocab_number[
            0] * num_w / float(num_token) + (1 - g_and_vocab_number[0]) / g_and_vocab_number[1]
    else:
        w_P = num_w / float(num_token)

    return w_P


def calc_entropy(input_list, corpus_list):
    """
    return the entropy of input_list as preprocessed corpus.
    input_list is the corpus to be calc
    corpus_list is the model corpus"""
    sum_token = sum([len(i) for i in corpus_list])
    sum_entropy = 0
    for i in input_list:
        for j in i:
            sum_entropy += - \
                (math.log2(calc_w_uni_P(j, corpus_list, 0.95, 100000)))
    return sum_entropy / sum_token


def calc_perplexity(input_entropy):
    return 2**input_entropy


def calc_coverage(input_list, model_list):
    model_vocab = set([set(i) for i in model_list])
    input_vocab = set([set(i) for i in input_list])
    return (input_vocab - model_vocab) / model_vocab


def unigram_model(model_list):
    vocab_type = set()
    for i in [set(j) for j in model_list]:
        vocab_type |= i
    for i in vocab_type:
        try:
            print("{}\t{}".format(i, calc_w_uni_P(i, model_list)))
        except UnicodeEncodeError:
            pass
            # input("press enter to continue")
    return

if __name__ == "__main__":
    if sys.argv[1]:
        model_list = preprocess_corpus(sys.argv[1])
        unigram_model(model_list)
    else:
        print("please enter a file name.")
