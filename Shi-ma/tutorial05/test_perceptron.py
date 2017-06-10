import train_perceptron

if __name__ == '__main__':
    epoch = 20
    path_data_train = '../../data/titles-en-train.labeled'
    w = train_perceptron.train_epoch(epoch, path_data_train)

    with open('../../data/titles-en-test.word', 'r') as data_test:
        with open('my_answer.txt', 'w') as data_out:
            for txt in data_test:
                txt = txt.lower()
                phi = train_perceptron.CREATE_FEATURES(txt)
                y_predict = train_perceptron.PREDICT_ONE(w, phi)
                print(y_predict, file=data_out)
