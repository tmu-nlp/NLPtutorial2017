from collections import defaultdict
from train_perceptron import predict_one, create_features
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
            yield ('{}\t{}'.format(y_,x))

if __name__ == '__main__':
    output_f = 'my_answer.labeled'
    model_f = 'model_file.txt'
    input_f = '../../data/titles-en-test.word'
    
    with open(output_f, 'w') as out_f:
        for text in predict_all(model_f, input_f):
            out_f.writelines(text)
