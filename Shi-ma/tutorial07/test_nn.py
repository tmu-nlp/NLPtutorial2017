import train_nn
import collections
import pickle

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

if __name__ == '__main__':
    ids = collections.defaultdict(lambda: 0)
    with open('train_nn_ids.result', 'r') as data_ids:
        for each_ids in data_ids:
            key, value = each_ids.split('\t')
            ids[key] = int(value)
            
    with open('train_nn_network.result', 'rb') as data_network:
        network = pickle.load(data_network)

    with open('../../data/titles-en-test.word', 'r') as data_test:
        with open('my_answer.txt', 'w') as data_out:
            for txt in data_test:
                txt = txt.lower()
                phi_0 = train_nn.CREATE_FEATURES(txt, ids, flag_test=1)
                phi = train_nn.FORWARD_NN(network, phi_0)
                predict_y = sign(phi[-1][0])
                print(predict_y, file=data_out)
