from train_svm import create_features, predict_one
from collections import defaultdict
import sys

inputfile = '../../data/titles-en-test.word'

def pred_all(model_file, input_file=inputfile):

    ansfile = model_file + '_ans'

    phi = defaultdict(lambda : 0.)
    with open(model_file) as modelf:
        w = defaultdict(lambda: 0.)
        for model_line in modelf:
            name, weight = model_line.split('\t')
            # print("{}\t{}".format(name,weight))
            w[name] = float(weight)

    with open(input_file) as inputf, open(ansfile, 'w') as ansout:
        for in_line in inputf:
            l = in_line
            # print(l)
            phi = create_features(l.lower())
            y_pred = int(predict_one(w, phi))
            print('{}\t{}'.format(y_pred, l), end='', file=ansout)

    import subprocess
    subprocess.call('../../script/grade-prediction.py ../../data/titles-en-test.labeled ' + ansfile, shell=True)

if __name__ == '__main__':
    modelfile = sys.argv[1]
    pred_all(modelfile,inputfile)
