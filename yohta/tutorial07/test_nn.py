import pickle
import numpy as np
from train_nn import forward_nn

def create_features_test(x): # UNI:word --> word_id
    # create list phi (len = len(ids))
    phi = np.zeros(len(ids))
    for word in x:
        # Training
        # phi[ids["UNI:" + word]] += 1
        # Testing
        if "UNI:" + word in ids:
            phi[ids["UNI:" + word]] += 1
    return phi

if __name__ == '__main__':
    with open('weight_file.txt','rb') as we_f,open('id_file.txt','rb') as id_f:
        net = pickle.load(we_f)
        ids = pickle.load(id_f)

    with open('../../data/titles-en-test.word','r') as i_f,open('my_answer.txt','w') as o_f:
        for line in i_f:
            x = line.split()
#            phi_0 = [0] * len(ids)
            phi_0 = create_features_test(x)
#        for line in i_f:
            phi = forward_nn(net,phi_0)
            if phi[len(net) - 1][0] >= 0:
                y_ = 1
            else:
                y_ = -1
            o_f.write('{}\n'.format(y_))
