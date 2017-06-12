from collections import defaultdict
import sys
import random


def create_features(sent):
    phi = defaultdict(lambda : 0)
    words = sent.strip().split(' ')
    for word in words:
        phi[word.lower()] += 1
    return phi


def get_weight(_w, _name, _c, _iter, _last):
    if _iter != _last[_name]:
        cumulative_c = _c * (_iter - _last[_name])
        if abs(_w[_name]) <= cumulative_c:
            _w[_name] = 0
        else:
            _w[_name] -= ((lambda x: 1 if x > 0 else -1)(_w[_name])) * cumulative_c
        last[_name] = _iter
    else:
        pass

    return _w[_name]


def w_dot_phi(_w, _phi):
    score = 0.
    for name, value in _phi.items():
        if name in _w.keys():
            score += value * _w[name]
    return score


def predict_one(_w, _phi):
    score = w_dot_phi(_w, _phi)
    if score >= 0.:
        return 1
    else:
        return -1


def eval_model(_modelfile):
    from test_svm import pred_all
    pred_all(_modelfile)


def dump(_epoch: int, total_epochs: int, _suffix):
    filename = './out/{}_of_{}_{}.model'.format(_epoch, total_epochs, _suffix)
    with open(filename, 'w') as model_f:
        for k,v in w.items():
            print('{}\t{}'.format(k,v), file=model_f)
    print('{}/{} finished.'.format(_epoch, total_epochs))
    eval_model(filename)


def check_type(var):
    print('{}'.format(type(var)))

if __name__ == '__main__':

    train_data = '../../data/titles-en-train.labeled'
    # train_data = 'titles-en-train.labeled_mini'

    n_iter = int(sys.argv[1])
    C = float(sys.argv[2])
    margin = float(sys.argv[3])
    suffix = '_'.join(sys.argv[2:5])

    w = defaultdict(lambda: 0.)

    # load sentence
    train_line_list = []
    with open(train_data) as f:
        for l in f:
            train_line_list.append(l.strip().split('\t'))

    for epoch in range(n_iter):
        if 'shff' in suffix:
            random.shuffle(train_line_list)
        for ll in train_line_list:
            y_label, sentence = int(ll[0]), ll[1]
            phi = create_features(sentence)
            val = w_dot_phi(w, phi) * y_label

            # print('val {}\tmargin{}'.format(val,margin))
            if val <= margin:
                for _iter, (name, v) in enumerate(phi.items()):
                    last = defaultdict(int)
                    w[name] = get_weight(w, name, C, _iter, last)
                    w[name] += v * y_label

        dump(epoch + 1, n_iter, suffix)
