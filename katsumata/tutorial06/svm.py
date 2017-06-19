from collections import defaultdict
train_f = '../../data/titles-en-train.labeled'
def predict_all(w, input_f):
    with open(input_f, 'r') as i_f:
        for x in i_f:
            phi = dict()
            for key, value in create_features(x).items():
                phi[key.replace('UNI:','')] = value
            y_ = sign(predict_one(w, phi))
            yield ('{}\t{}'.format(y_,x))
#一事例に対する予測            
def predict_one(w, phi):
    score = 0
    #ドット積と等価
    for name, value in phi.items():
        if name in w:
            score += value*w[name]
    return score 
#sign作成
def sign(score):
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
def update_weights(w, phi, y, c, iteration, last):
    """
    for name, value in w.items():
        if abs(value) < c:
            w[name] = 0
        else:
            w[name] -= sign(value) * c
    """        
    for name, value in phi.items():
        w[name] = getw(w, name, c, iteration, last)
        w[name] += int(value)*y
#重みの正則化
def getw(w, name, c, iteration, last):
    if iteration != last[name]:     #重みが古くなっていたら更新するため速い！
        c_size = c*(iteration - last[name])
        if abs(w[name]) <= c_size:
            w[name] = 0
        else:
            w[name] -= sign(w[name]) * c_size
        last[name] = iteration
    return w[name]    

if __name__ == '__main__':
    output_f = 'my_answer.test'
    input_f = '../../data/titles-en-test.word'
    w = defaultdict(int)
    last = defaultdict(int)
    l = 20 #iteration
    c = 10 ** -5
    margin = 1
    for i in range(l):
        with open(train_f, 'r') as t_f:
            for line in t_f:
                y, x = line.strip().split('\t')
                phi = dict()
                for key, value in create_features(x).items():
                    phi[key.replace('UNI:', '')] = value
                val = predict_one(w, phi) * int(y)
                if val <= margin:
                    update_weights(w, phi, int(y), c, i, last)
    with open(output_f, 'w') as out_f:
        for text in predict_all(w, input_f):
            out_f.writelines(text)
    """                
    with open(model_f, 'w') as m_f:                
        for key, value in w.items():
            #print ('{} {}'.format(key, value))
            m_f.write('{} {}\n'.format(key, value)) 
    """ 
