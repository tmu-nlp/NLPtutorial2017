#from collections import defaultdict
import sys
import math
#probabilities = defaultdict(lambda: 0)
"""
W:トークン数
V:未知語を含む語彙数
unk:未知語数
"""
lambda1 = .95
lambda_unk = 1 - lambda1
V = 1000000
W = 0
H = 0
unk = 0
probabilities = dict()
#モデル読み込み
with open('model_file.txt', 'r') as m_f:
    for row in m_f:
        words = row.split()
        probabilities[words[0]] = float(words[2])
#評価と結果表示
with open(sys.argv[1], 'r') as test_file:
    for row in test_file:
        words = row.split()
        #append </s>to eof
        words.append('</s>')
        for word in words:
            W += 1
            P = lambda_unk / V
            word_small = word.lower()
            if word_small in probabilities:
                P += lambda1 * probabilities[word_small]
            else:
                unk += 1
#add 1 to unk
            H += -1 * math.log(P, 2)
print ('entropy =  %f' %(H/W))
print ('coverage =  %f' %((W - unk)/ W))
