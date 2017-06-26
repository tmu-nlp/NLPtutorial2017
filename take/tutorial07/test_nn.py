from _train_nn import forward_nn
import numpy as np
import pickle

def create_features(x):
    phi = [0] * len(ids)
    for word in x:
        try:
            phi[ids[word]] += 1
        except KeyError:
            pass
    return phi

def pred(model_file:str):
    
    # ids = dict()
    with open(model_file,'rb') as model_f, open('ids.dat') as ids_f:
        nn = pickle.load(model_f)
        for line in ids_f:
            line = line.strip().split('\t')
            ids[line[0]] = int(line[1])
    
    ansfile='ans'
    with open('../../data/titles-en-test.word') as f, open(ansfile, 'w') as ans_f:
        for line in f :
            words = line.split()
            phi0 = create_features(words)
            phi = forward_nn(nn, phi0)
            predicted = 1 if phi[len(nn) - 1][0] >= 0 else -1
            print(predicted, file=ans_f)

    import subprocess
    subprocess.call('../../script/grade-prediction.py ../../data/titles-en-test.labeled ' + ansfile, shell=True)


if __name__ == '__main__':

    import sys
    model = sys.argv[1]
    ids = dict()
    pred(model)    

