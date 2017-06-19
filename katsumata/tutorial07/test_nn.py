import pickle
from train_nn import forward_nn

def create_features(x):
    phi = [0 for i in range(len(ids))]
    words = x.strip().split()
    for word in words:
        if 'UNI:'+word in ids:
            phi[ids['UNI:'+word]] += 1
    return phi        

if __name__ == '__main__':
    test_f = '../../data/titles-en-test.word'
    answer = 'my_answer.nn' 
    with open('network.dump', 'rb') as net_f:
        net = pickle.load(net_f)
    with open('ids.dump', 'rb') as ids_f:
        ids = pickle.load(ids_f)
    with open(answer, 'w') as ans_f, open(test_f, 'r') as t_f:
        for x in t_f:
            x = x.strip()
            phi_0 = create_features(x)
            phiS = forward_nn(net, phi_0)
            y_ = (1 if phiS[len(net)-1][0] >= 0 else -1)
            ans_f.write ('{}\t{}\n'.format(y_, x))
