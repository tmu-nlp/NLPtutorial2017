
# coding: utf-8

# In[9]:

import numpy as np
import pickle
from pprint import pprint
import pdir


# In[2]:

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# In[25]:

def gen_ids(f_name):
    xs,ys=[],[]
    
    for line in open(f_name,"r",encoding="utf-8"):
        for x_y in line.split():
            x,y=x_y.split("_")
            xs.append(x)
            ys.append(y)
    
    x_set=set(xs)
    y_set=set(ys)
    
    x_ids,y_ids={},{}
    for x in x_set:
        x_ids[x]=len(x_ids)
    for y in y_set:
        y_ids[y]=len(y_ids)
        
    return x_ids,y_ids


# In[30]:

def gen_array(x_y_sentence,x_ids,y_ids):
    len_sentence=len(x_y_sentence.split())
    x_array=np.zeros((len_sentence,len(x_ids)))
    y_array=np.zeros((len_sentence,len(y_ids)))
    for idx, x_y in enumerate(x_y_sentence.split()):
        x,y =x_y.split("_")
        x_array[idx][x_ids[x]]=1
        y_array[idx][y_ids[y]]=1
    return x_array,y_array


# In[31]:

def gen_arrays(x_y_file,x_ids,y_ids):
    x_arrays,y_arrays=[],[]
    for line in open(x_y_file,"r",encoding="utf-8"):
        x_array,y_array=gen_array(line,x_ids,y_ids)
        x_arrays.append(x_array)
        y_arrays.append(y_array)
    return x_arrays,y_arrays


# In[32]:

x_ids,y_ids=gen_ids("../../test/05-train-input.txt")
print(x_ids,y_ids)


# In[36]:

x_arrays,y_arrays=gen_arrays("../../test/05-train-input.txt",x_ids,y_ids)
pprint(x_arrays)
pprint(y_arrays)


# In[37]:

pickle.dump((x_ids,y_ids),open("w_y_ids.pkl","wb"))
pickle.dump((x_arrays,y_arrays),open("x_y_arrays.pkl","wb"))


# In[ ]:

class NN():

    def __init__(self, inputs, network_dims):
        assert len(h_prev) == len(x)
        
        self.h_prev = input_list
        self.x = x
        
        self.network_neuron_nets = []
        self.network_neuron_outputs = []
        self.network_neuron_bias = []

        self.network_neuron_derr_net = []
        self.network_neuron_dnet_weight_x = []
        self.network_neuron_dnet_weight_h = []

        self.network_weights_x = []
        self.network_weights_h = []

        dim_prev = len(h_prev)
        for i in range(len(network_dims)):
            dim = network_dims[i]
            self.network_weights_x.append(np.random.rand(dim_prev, dim) / 10)
            self.network_weights_h.append(np.random.rand(dim_prev, dim) / 10)
            self.network_neuron_bias.append(np.random.rand(dim) / 10)
            self.network_neuron_outputs.append(np.zeros((dim)))
            self.network_neuron_nets.append(np.zeros((dim)))
            self.network_neuron_derr_net.append(np.zeros((dim)))
            self.network_neuron_dnet_weight_x.append(np.zeros((dim)))
            self.network_neuron_dnet_weight_h.append(np.zeros((dim)))
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


# In[11]:

class RNN_unit():
    def __init__(self,h_prev,x):
        assert len(h_prev)==len(x)
        self.h_prev=h_prev
        self.x=x
        
        self.w_rh=np.random.rand((1,len(x_ids)))
        self
        


# In[38]:

h_prev=np.random.rand(len(x_ids))
x=np.random.rand(len(x_ids))
rnn_unit=RNN_unit(h_prev,x)


# In[40]:

rnn_unit.x


# In[ ]:

rnn_unit=RNN_unit()


# In[68]:

class RNN():
    def __init__(self,Xs,Ys):
        self.Xs=Xs
        self.Ys=Ys
        self.w_rx=np.random.rand(len(Xs))
        self.w_rh=np.random.rand(len(Xs))
        self.b_r=np.random.rand(len(Xs))
        self.w_oh=np.random.rand(len(Xs))
        self.b_o=np.random.rand(len(Xs))
        self.net=(self.w_rx,self.w_rh,self.b_r,self.w_oh,self.b_o)
    def find_best(self,ps):
        assert isinstance(ps,list)
        y=0
        for idx, i in enumerate(ps):
            if ps[y]<ps[i]:
                y=i
        return y
    def create_one_hot(self,hot_id,size):
        assert isinstance(hot_id,int)
        assert isinstance(size,int)
        vec=np.zeros(size)
        vec[hot_id]=1
        return vec
    def forward_rnn(self,w_rx,w_rh,b_r,w_oh,b_o,x):
        assert w_rx.shape[1]==x.shape[0]
        h=[]
        p=[]
        y=[]
        for t in range(len(x)):
            if t>0:
                h[t]=np.tanh(np.matmul(w_rx,x[t])+np.matmul(w_rh,h[t-1])+b_r)
            else:
                h[t]=np.tanh(np.matmul(w_rx,x[t])+b_r)
            p[t]=softmax(np.tanh(w_oh,h[t]))
            y[t]=self.find_best(p[t])
        return h,p,y
    def gradient_rnn(self,w_rx,w_rh,b_r,w_oh,b_o,x,h,p,y_):
        dw_rx=0
        dw_rh=0
        db_r=0
        dw_oh=0
        db_o=0
        
        err_r_=np.zeros(len(b_r))
        
        for t in reversed(range(len(x))):
            p_=self.create_one_hot(y_[t])
            err_o_=p_-p[t]
            dw_oh+=np.outer(h[t],err_o_)
            db_o+=err_o_
            err_r=np.dot(err_r_,w_rh)+np.dot(err_o_,w_oh)
            err_r_=err_r*(1-h[t]**2)
            dw_rx+=np.outer(x[t],err_r_)
            db_r+=err_r_
            if t!=0:
                dw_rh+=np.outer(h[t-1],err_r_)
        return dw_rx,dw_rh,db_r, dw_oh, db_o
    def update_weights(self,w_rx,w_rh,b_r,w_oh,b_o,dw_rx,dw_rh,db_r,dw_oh,db_o,lrate=0.01):
        w_rx+=lrate*dw_rx
        w_rh+=lrate*dw_rh
        b_r+=lrate*db_r
        w_oh+=lrate*dw_oh
        b_o+=lrate*db_o
        return w_rx, w_rh,b_r,w_oh,b_o
    def train(self):
        for i in range(10):
            for x,y_ in zip(self.Xs,self.Ys):
                h,p,y=self.forward_rnn(*self.net,x)
                delta=self.gradient_rnn(*self.net,x,h,y_)
                self.w_rx,self.w_rh,self.b_r,self.w_oh,self.b_o=self.update_weights(*self.net,*delta,lrate=0.1)
        print(*self.net, file=open("weight.file","w"))


# In[ ]:




# In[69]:

rnn=RNN(x,y)


# In[70]:

rnn.net


# In[71]:

rnn.train


# In[72]:

rnn.net


# In[ ]:



