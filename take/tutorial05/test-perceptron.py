from train_perceptron import create_features, predict_one
from collections import defaultdict

def pred_all(model_file, input_file):

    phi = defaultdict(lambda : 0)
    with open(model_file) as modelf:
        w = defaultdict(lambda: 0)
        for model_line in modelf:
            name, weight = model_line.split('\t')
            # print("{}\t{}".format(name,weight))
            w[name] = int(weight)

    with open(input_file) as inputf:
        for in_line in inputf:
            l = in_line
            # print(l)
            phi = create_features(l.lower())
            y_pred = int(predict_one(w,phi))
            print('{}\t{}'.format(y_pred, l), end='')

# modelfile = 'iter_1.model'
modelfile = 'iter_10_shff.model'
inputfile = '../../data/titles-en-test.word'

if __name__ == '__main__':
    pred_all(modelfile,inputfile)

    # import subprocess
    # subprocess.call('../../script/grade-prediction.py ../../data/titles-en-test.labeled '+ outfile, shell=True)
    # ../../script/grade-prediction.py ../../data/titles-en-test.labeled 20.ans
