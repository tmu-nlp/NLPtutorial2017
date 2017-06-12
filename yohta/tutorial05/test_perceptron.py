from collections import defaultdict
from train_perceptron import predict_one, create_features

def predict_all(model_file,input_file):
    with open(model_file,'r') as m_f,open(input_file,'r') as i_f:
        w = defaultdict(lambda :0)
        for line in m_f:
            word,value = line.split('\t')
            w[word] = int(value)
        for x in i_f:
            phi = defaultdict(lambda :0)
            for k,v in create_features(x).items():
                phi[k] = int(v)
            y_ = predict_one(w,phi)
            yield('{}\t{}'.format(y_,x))

if __name__ == '__main__':
    with open('my_answer.labeled','w') as ans:
        for line in predict_all('model_file.txt','../../data/titles-en-test.word'):
            ans.write(line)
