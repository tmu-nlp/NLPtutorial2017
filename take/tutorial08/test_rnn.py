from collections import defaultdict
import numpy as np
import dill
from train_rnn import *


if __name__ == '__main__':

    ansfile = 'ans.out'
    
    with open(model_file,'rb') as f:
        x_ids, y_ids, net = dill.load(f)

    # 品詞idからPOSを引くため
    y_id2pos = {v:k for k, v in y_ids.items()}
    
    # test_data = "../../data/wiki-en-test.norm"
    test_data = "../../test/05-test-input.txt"
    
    with open(test_data) as f, open(ansfile, 'w') as ans_f:
        for line in f:
            _w = list()
            words = line.strip().lower().split(" ")
            for word in words:
                if word in x_ids:
                    _w.append(create_one_hot(len(x_ids), x_ids[word]))
                else:
                    '''unknown word'''
                    pass
    
            _, _, pred_tag_id = forward_rnn(net, _w)
            # print(pred_tag_id)
            pred_tag_name = [y_id2pos[x] for x in pred_tag_id]
            print(' '.join(pred_tag_name), file=ans_f)
