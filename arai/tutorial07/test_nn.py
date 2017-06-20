from collections import defaultdict
import pickle
import train_nn 

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

def create_features(x):
    phi0 = [0] * len(ids)
    words = x.split()
    for word in words:
        if word in ids:
            phi0[ids[word]] += 1
    return phi0
    

if __name__ == '__main__':
    with open('id_file.txt') as id_f:
        ids = {}
        for iddayo in id_f:
            x, y = iddayo.split()
            ids[x] = int(y)
    with open('weight_file.txt','rb') as w_f:
        weight = pickle.load(w_f)

    with open('../../data/titles-en-test.word') as  text:
        with open('my_answer.txt', 'w') as answer:
            for line in text:
                phi0 = create_features(line)
                phi = train_nn.forward_nn(weight, phi0)
                ydash = sign(phi[-1][0])
                print(ydash, file = answer)

