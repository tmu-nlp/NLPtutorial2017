from collections import defaultdict
train_f = '../../data/titles-en-train.labeled'
model_f = 'model_file.txt'
"""
def predict_all(model_f, input_f):
    w = defaultdict(int)
    with open(model_f, 'r') as m_f, open(input_f, 'r') as i_f:
        #モデルファイルから各単語の重みを読み込む
        for line in m_f:
            words = line.strip().split()
            w[words[0]] = int(words[1]) 
        for x in i_f:
            #phi = create_features(x)
            phi = dict()
            for key, value in create_features(x).items():
                phi[key.replace('UNI:','')] = value
            y_ = predict_one(w, phi)
            #yield str(y_)+'\t'+x
            #print (x.strip('\n'))
            print ('{}\t{}'.format(y_,x.strip('\n')))
"""
#一事例に対する予測            
def predict_one(w, phi):
    score = 0
    #ドット積と等価
    for name, value in phi.items():
        if name in w:
            score += value*w[name]
    if score >= 0:
        return 1
    else:
        return -1
#素性作成    
def create_features(x):
    phi = defaultdict(int)
    words = x.strip().split()
    for word in words:
        phi['UNI:'+word] += 1
    return phi
#重みの更新    
def update_weights(w, phi, y):
    for name, value in phi.items():
        w[name] += int(value)*y

if __name__ == '__main__':
    w = defaultdict(int)
    l = 10 #iteration
    for i in range(l):
        with open(train_f, 'r') as t_f:
            for line in t_f:
                y, x = line.strip().split('\t')
                phi = dict()
                for key, value in create_features(x).items():
                    phi[key.replace('UNI:', '')] = value
                y_ = predict_one(w, phi)
                if y_ != int(y):
                    update_weights(w, phi, int(y))
    with open(model_f, 'w') as m_f:                
        for key, value in w.items():
            #print ('{} {}'.format(key, value))
            m_f.write('{} {}\n'.format(key, value))
