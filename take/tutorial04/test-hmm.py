from collections import defaultdict
from pprint import pprint
from math import log2

# unknown smoothing
lambda_coef = 0.95
lambda_unk = 1. - lambda_coef
V = 10**6


emission = defaultdict(lambda : lambda_unk/V) # initialize with lambda param.
transition = defaultdict(float)
possible_tags = dict()

# model_file = 'test.model'
# test_input = '../../test/05-test-input.txt'

model_file = 'wiki.model'
test_input = '../../data/wiki-en-test.norm'

# train_input = '../../test/05-train-input.txt'
# train_answer = '../../test/05-train-answer.txt'
# test_input = './05-test-input.txt'
# test_answer = '../../test/05-test-answer.txt'

# load model
with open(model_file, "r") as f:
    for line in f:
        _type, _ctx, _word, _prob = line.strip().split(' ')
        # print("type:{} ctx:{} word:{} prob:{}".format(_type, _ctx, _word, _prob))
        possible_tags[_ctx] = 1
        if _type == 'T':
            transition[_ctx + ' ' + _word] = float(_prob)
            # pprint("T: {}, p={}".format(_ctx + ' -> ' + _word, _prob))
        else:
            emission[_ctx + ' ' + _word] = float(_prob)
            # pprint("E: {}, p={}".format(_ctx + ' -> ' + _word, _prob))


# pprint(emission)
# pprint(transition)
# pprint(possible_tags)


with open(test_input) as f:
    # Step Forword
    for l in f:
        words_list = l.strip().split(' ')
        # print(words_list)
        best_score = defaultdict(float)
        best_edge = defaultdict(str)
        best_score['0 <s>'] = 0.
        best_edge['0 <s>'] = None
        num_of_words = len(words_list)
        for i in range(num_of_words): # 0 <= i < len(word)
            for prev_tag in possible_tags.keys():
                for next_tag in possible_tags.keys():
                    if (str(i) + ' ' + prev_tag) in best_score.keys() and \
                            (prev_tag + ' ' + next_tag) in transition.keys():
                        score = best_score[str(i) + ' ' + prev_tag] \
                                - log2(lambda_coef * emission[next_tag + ' ' + words_list[i]]) \
                                - log2(transition[prev_tag + ' ' + next_tag])
                        # print('E->',lambda_coef * emission[next_tag + ' ' + words_list[i]])
                        if str(i+1) + ' ' + next_tag not in best_score.keys() \
                                or best_score[str(i+1) + ' ' + next_tag] > score:
                            best_score[str(i+1) + ' ' + next_tag] = score
                            best_edge[str(i+1) + ' ' + next_tag] = str(i) + ' ' + prev_tag

        #最後に終端記号</s>への遷移を計算する
        # i -> num_of_words
        next_tag = '</s>'
        for prev_tag in possible_tags.keys():
            if (str(num_of_words) + ' ' + prev_tag) in best_score.keys() and \
                            (prev_tag + ' ' + next_tag) in transition.keys():
                score = best_score[str(num_of_words) + ' ' + prev_tag] \
                        - log2(transition[prev_tag + ' ' + next_tag])
                if str(num_of_words + 1) + ' ' + next_tag not in best_score.keys() \
                        or best_score[str(num_of_words + 1) + ' ' + next_tag] > score:
                    best_score[str(num_of_words + 1) + ' ' + next_tag] = score
                    best_edge[str(num_of_words + 1) + ' ' + next_tag] = str(num_of_words) + ' ' + prev_tag

        #Backword
        tags = []
        next_edge = best_edge[str(num_of_words + 1) + ' </s>']
        while next_edge is not None:
            # print(next_edge)
            pos, tag = next_edge.split(' ')
            if tag == "<s>":
                break
            tags.append(tag)
            next_edge = best_edge[next_edge]
        tags.reverse()
        print(' '.join(tags))