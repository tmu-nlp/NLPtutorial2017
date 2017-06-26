
# coding: utf-8

# In[1]:

import numpy as np
from IPython.core.debugger import Tracer
#this one triggers the debugger
from collections import defaultdict
from pprint import pprint


# In[2]:

idx_dict=defaultdict(lambda:len(idx_dict))
phi_dict=defaultdict(int)


# In[3]:

def update_phi(line,isupdate=True):
    if isupdate:
        for w in line.split():
            phi_dict[idx_dict[w]]+=1


# In[4]:

def get_lines_from_file(f_name):
    for line in open(f_name, "r",encoding="utf-8"):
        yield line


# In[5]:

for line in get_lines_from_file("../../test/03-train-input.txt"):
    cls, line=line.split("\t")
    update_phi(line)


# In[6]:

pprint(idx_dict)
phi_array=np.array(phi_dict.values())
pprint(phi_array)
assert len(idx_dict)==len(phi_dict)


# In[7]:

def gen_training_phi(line):
    phi_array_ = np.zeros(len(phi_dict))
    for w in line.split():
        phi_array_[idx_dict[w]] += 1
    return phi_array_


# In[8]:

for line in get_lines_from_file("../../test/03-train-input.txt"):
    cls, line=line.split("\t")
    print(gen_training_phi(line))


# In[9]:

def initialize_networks(*num_neurons_network):
    num_neurons_prev=len(idx_dict)
    ws_network=[]
    b=np.array(1.0)
    
    for num_neurons_current in num_neurons_network:
        ws_layer=[]
        for n in range(num_neurons_current):
            ws_neuron=np.random.rand(num_neurons_prev)
            ws_layer.append((ws_neuron,b))
        ws_network.append(ws_layer)
        num_neurons_prev=len(ws_layer)
    
    return ws_network
        


# In[10]:

ws_network=initialize_networks(2,1)
pprint(ws_network)


# In[11]:

def forward_nn(ws_network,phi_input):
    phis_prev=phi_input
    phis_network=[phi_input]
    #pprint(ws_network)
    for idx_layer,ws_layer in enumerate(ws_network):
        #pprint(ws_layer)
        phis_layer=[]
        for idx_neuron,ws_neuron in enumerate(ws_layer):
            #pprint(ws_neuron)
            phis_layer.append(np.tanh(np.dot(phis_prev,ws_neuron[0])+ws_neuron[1]))
        phis_prev=phis_layer
        phis_network.append(np.array(phis_layer))
    return phis_network


# In[12]:

phis_network=forward_nn(ws_network,gen_training_phi("Shoken , monk born in Kyoto"))
pprint(phis_network)


# In[13]:

def backward_nn(ws_network,phis_network,label):
    grad_deltas_network=[]
    deltas_network=[]
    
    for idx_layer, layer in enumerate(reversed(phis_network)):
        
        if idx_layer==0:
            delta=np.array(label - layer)
            #print(delta)
            last_delta=delta
            last_layer=layer
            
        else:
            grad_delta=last_delta*(1-last_layer**2)
            weight=np.vstack([t[0] for t in ws_network[-idx_layer]])
            delta=np.dot(grad_delta,weight)
            #print(grad_delta,delta)
            last_delta=delta
            last_layer=layer
            #deltas_layer.append(delta)
            #grad_deltas_layer.append(grad_delta)
            deltas_network.append(delta)
            grad_deltas_network.append(grad_delta)
    
    return grad_deltas_network


# In[14]:

grad_deltas=backward_nn(ws_network,phis_network,1)
pprint(grad_deltas)


# In[15]:

def update_weights(ws_network, phis_network, grad_deltas, learning_rate=0.01):
    new_ws_network=[]
    for idx_layer, layer in enumerate(phis_network):
        new_ws_layer=[]
        if idx_layer == 0:
            layer_prev = layer
        else:
            grad_weight = np.outer(grad_deltas[-idx_layer], layer_prev)
            layer_prev = layer
            weight=np.vstack([w[0] for w in ws_network[idx_layer-1]])
            bias=np.hstack([w[1] for w in ws_network[idx_layer-1]])
            weight+=learning_rate*grad_weight
            bias+=learning_rate*grad_deltas[-idx_layer]
            for i ,j in zip(weight, bias):
                new_ws_layer.append((np.array(i),np.array(j)))
            new_ws_network.append(new_ws_layer)
    return new_ws_network


# In[16]:

update_weights(ws_network,phis_network,grad_deltas)


# In[17]:

def train_model(f_name,ws_network,num_iteration=10000):
    for i in range(num_iteration):
        for line in get_lines_from_file("../../test/03-train-input.txt"):
            cls, line=line.split("\t")
            cls=int(cls)
            phi=gen_training_phi(line)
            phi=forward_nn(ws_network,phi_input=phi)
            grad_delta=backward_nn(label=cls,phis_network=phi,ws_network=ws_network)
            ws_network=update_weights(ws_network,phis_network=phi,grad_deltas=grad_delta)
    return ws_network


# In[18]:

train_model("../../test/03-train-input.txt",ws_network)


# In[19]:

forward_nn(ws_network,gen_training_phi("Shoken , monk born in Kyoto"))


# In[ ]:



