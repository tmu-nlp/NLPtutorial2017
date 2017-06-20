import pickle

from train_nn import FORWORD_NN

def CREATE_FEATURES(x):
    phi = [0 for i in range(len(ids))]
    words = x.strip().split()
    for word in words:
        if 'UNI:'+word in ids:
            phi[ids['UNI:'+word]] += 1
    return phi

if __name__ == '__main__':
    with open('network.dump', 'rb') as net_f:
        net = pickle.load(net_f)
    with open('ids', 'rb') as ids_f:
        ids = pickle.load(ids_f)

    with open('my_answer.txt', 'w') as ans_f, open('../../data/titles-en-test.word', 'r') as t_f:
        for x in t_f:
            x = x.strip()
            x_l = x.lower()
            phi_0 = CREATE_FEATURES(x_l)
            phi = FORWORD_NN(net, phi_0)
            y = (1 if phi[len(net)][0] >= 0 else -1)
            ans_f.write ('{}\t{}\n'.format(y, x))
