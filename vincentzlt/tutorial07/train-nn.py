
# coding: utf-8

# In[226]:

import numpy as np
from pprint import pprint
from time import time
import pdir


# In[209]:

def gen_ids(f_names):
    words=[]
    labels=[]
    for line in open(f_names,"r",encoding="utf-8"):
        label,sentence=line.split("\t")
        labels.append(int(label))
        for w in sentence.split():
            words.append(w)
    words=set(words)
    w_ids={}
    
    for w in words:
        w_ids[w]=len(w_ids)
    sentences=[]
    for line in open(f_names,"r",encoding="utf-8"):
        label,sentence=line.split("\t")
        sent=np.zeros(len(w_ids))
        for w in sentence.split():
            sent[w_ids[w]]+=1
        sentences.append(sent)
    return w_ids,sentences,labels


# In[249]:

def gen_sent_phi(sentence,w_ids):
    sent=np.zeros(len(w_ids))
    for w in sentence.split():
        if w in w_ids:
            sent[w_ids[w]]+=1
    return sent


# In[211]:

w_ids,sentences,labels=gen_ids("../../data/titles-en-train.labeled")


# In[212]:

sentences


# In[213]:

labels


# In[243]:

class NN():

    def __init__(self, input_list, labels,w_ids, network_dims):
        assert len(input_list) == len(labels)
        
        self.w_ids=w_ids
        
        self.input_list = input_list
        self.labels = labels
        self.network_neuron_outputs = []
        self.network_neuron_nets = []
        self.network_neuron_bias = []

        self.network_neuron_derr_net = []
        self.network_neuron_dnet_weight = []

        self.network_weights = []

        dim_prev = len(input_list[0])
        for i in range(len(network_dims)):
            dim = network_dims[i]
            self.network_weights.append(np.random.rand(dim_prev, dim) / 10)
            self.network_neuron_bias.append(np.random.rand(dim) / 10)
            self.network_neuron_outputs.append(np.zeros((dim)))
            self.network_neuron_nets.append(np.zeros((dim)))
            self.network_neuron_derr_net.append(np.zeros((dim)))
            self.network_neuron_dnet_weight.append(np.zeros((dim)))
            dim_prev = dim

        self.network = (self.network_neuron_outputs, self.network_neuron_nets,
                        self.network_neuron_derr_net,
                        self.network_neuron_dnet_weight)

    def print_network(self):
        for w in self.__dict__:
            if w.startswith("network"):
                print(w)
                pprint(self.__dict__[w])
                print()

    def ff_one(self, input_array):
        self.input_array = input_array
        for idx in range(len(self.network_weights)):
            if idx == 0:
                outputs_prev = input_array
            else:
                outputs_prev = self.network_neuron_outputs[idx - 1]
            self.network_neuron_nets[idx] = np.dot(
                outputs_prev,
                self.network_weights[idx]) + 1 * self.network_neuron_bias[idx]
            self.network_neuron_outputs[idx] = np.tanh(
                self.network_neuron_nets[idx])

    def bk_one(self, label):
        err = label - self.network_neuron_outputs[-1]
        for i in reversed(range(len(self.network_neuron_outputs))):
            out = self.network_neuron_outputs[i]
            net = self.network_neuron_nets[i]
            if i != 0:
                out_prev = self.network_neuron_outputs[i - 1]
            else:
                out_prev = self.input_array
            w = self.network_weights[i]
            if i == len(self.network_neuron_outputs) - 1:
                derr_out = label - self.network_neuron_outputs[-1]
            else:
                derr_out = np.dot(self.network_neuron_derr_net[i + 1],
                                  self.network_weights[i + 1].T)
            dout_net = 1 - out**2
            

            self.network_neuron_derr_net[i] = derr_out * dout_net
            dnet_weight = np.outer(out_prev, self.network_neuron_derr_net[i])

            self.network_neuron_dnet_weight[i] = dnet_weight

    def update_weight(self, lrate=0.01):
        for i in range(len(self.network_weights)):
            self.network_weights[i] += self.network_neuron_dnet_weight[
                i] * lrate
            self.network_neuron_bias[i] += self.network_neuron_derr_net[
                i] * lrate

    def train(self, epoch=10, lrate=0.01):
        for i in range(epoch):
            start=time()
            print("Epoch start:\t{}".format(i+1))
            for input_array, label in zip(self.input_list, self.labels):
                self.ff_one(input_array)
                self.bk_one(label)
                self.update_weight(lrate=lrate)
            print("Epoch ends:\t{}. Spending {:0.2f} seconds.\n".format(i+1,time()-start))
    def predict(self,sent):
        self.ff_one(gen_sent_phi(sent,self.w_ids))
        cls=self.network_neuron_outputs[-1]
        if cls>0:
            return 1
        else:
            return -1
            


# In[244]:

nn=NN(sentences,labels,w_ids,(2,1))


# In[245]:

nn.train(epoch=10,lrate=0.1)


# In[251]:

test_file="../../data/titles-en-test.word"
with open("my-answer","w",encoding="utf-8") as f:
    for sent in open(test_file,"r",encoding="utf-8"):
        f.write("{}\t{}".format(nn.predict(sent),sent))


# In[ ]:



