from collections import defaultdict

def sign(w):#y=sign(w⋅ϕ( x))
    if w >=0:
        return 1
    else:
        return 0

def create_feature(x):
    phi = defaultdict(int)
    words =x.split()
    for word in words:
        phi["UNI:{}".format(word)] += 1 #We add “UNI:” to indicate unigrams
    return phi


def update(w,name,y,value, val, c=0.001):
    if abs(w[name]) < c:
        w[name] = 0
    else:
        w[name] -= sign(val) * c
    w[name] += value * float(y)

def train(w, margin=120.0):
    for line in open(input_file):
        y_, x = line.split("\t")
        y = float(y_)
        phi = create_feature(x)
        #point! merge loop phi.items() stuff into one, if not; gonna be heavy mak
        for name, value in phi.items():# if name not in phi, by default = 0
            val = w[name] * value * y
            if val <= margin:
                update(w,name,y,value,val)

if __name__ == "__main__":
    input_file = "../../data/titles-en-train.labeled"
    #input_file = "../test/03-train-input.txt"
    w = defaultdict(int)
    l_iter = 20
    for i in range(l_iter):
        train(w)
    with open("model.txt", "w") as output:
        for name, value in w.items():
            print("{}\t{}".format(name,value),file=output)
