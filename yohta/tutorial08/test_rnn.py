from train_rnn import *
import pickle
import numpy as np

if __name__ == '__main__':
    with open('weight_file','rb') as w_f,open('id_file','rb') as id_f:
        rnn_net = pickle.load(w_f)
        x_ids,y_ids,word_data,pos_data = pickle.load(id_f)

#    x_list = []
    with open('../../data/wiki-en-test.norm') as test,open('my_answer','w') as o_f:
        output = []
        for line in test:
            ans = []
            x_list = []
            words = line.split()
            for word in words:
                if word in word_data:
#                    print(1)
                    x_list.append(create_onehot(x_ids[word],len(x_ids)))
                else: # 未知語の処理が分からん
                    x_list.append(np.zeros(len(x_ids)))
#            print(x_list)
            h,p,y_l = forward_rnn(rnn_net,x_list)
#        print(sorted(y_ids.items(),key = lambda x:x[1]))
            for i in y_l:
    #            print(sorted(y_ids.items(),key = lambda x:x[1])[i][0])
                ans.append(sorted(y_ids.items(),key = lambda x:x[1])[i][0])
        #    print(ans)
        #    answer = ' '.join(ans)
        #    output.append(answer)
    #        print(len(ans))
    #        break
            answer = ' '.join(ans)
#            print(answer)
            print(answer,file = o_f)
        #o_f.write('{}'.format(answer))
    #        ans = [pos_data[i] for i in y_l]
    #        print(ans)
    #            print(y_a)
        #        for i in range(len(y_ids)):
        #            if y_a[i] == 1:
        #                index = counter
        #            counter += 1
        #        print(index)
#                ans.append(y_ids[index])
#            print(' '.join(ans))
#            for y_unk in y_l:
