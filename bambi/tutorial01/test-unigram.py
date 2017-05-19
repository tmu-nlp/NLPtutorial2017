import math
from collections import defaultdict
probabilities = defaultdict(lambda: 0)

with open("model.word", "r") as model:
    for line in model.readlines():
        words = line.split()
        probabilities[words[0]] = words[1]

def test(file_path):
    lam_1 = 0.95
    lam_unknown = 1 - lam_1
    V = 1000000
    W = 0
    H = 0
    unknown = 0
    with open(file_path, "r") as test_file:
        for line in test_file.readlines():
            words = line.split()
            words.append("</s>")
            for word in words:
                W += 1
                P = lam_unknown / V
                if probabilities[word]:
                    P += lam_1 * float(probabilities[word])#probabilities[word] is str, need to cast for calculation
                else:
                    unknown += 1
                H -= math.log(P,2)
    print(file_path)
    print("entropy = {}".format(H/W))
    print("coverage = {}".format((W-unknown)/W))

#test("../../test/01-train-input.txt")
#test("../../test/01-test-input.txt")
test("../../data/wiki-en-test.word")
