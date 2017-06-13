#  result
#  non-shuffle epoch 01 90.967056%
#  non-shuffle epoch 10 93.446688%
#  non-shuffle epoch 20 93.234148%
#  non-shuffle epoch 30 92.950762%
#  shuffle     epoch 01 91.604676%
#  shuffle     epoch 10 93.269571%
#  shuffle     epoch 20 93.765498%
#  shuffle     epoch 30 93.375841%


import importlib
train_perceptron = importlib.import_module('train-perceptron')
train = train_perceptron.train
create_features= train_perceptron.create_features
predict_one = train_perceptron.predict_one
import sys

def test(test_file, weight):
    with open(test_file) as f:
        for line in f:
            sent = line.strip()
            phi = create_features(sent)
            y, weight = predict_one(weight, phi)
            print(y)

if __name__ == "__main__":
    w = train(sys.argv[1])
    test(sys.argv[2], w)
