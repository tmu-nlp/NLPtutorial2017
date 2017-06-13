
# coding: utf-8

# In[29]:

import sys,math
from collections import defaultdict
from pprint import pprint


# In[108]:

def sign(number):
    if number>=0:
        return 1
    else:
        return -1


# In[116]:

class SGD():
    def __init__(self, f_name, iter_times_training=10):
        labels = []
        lines = []
        self.ws = defaultdict(int)

        # read lines and labels
        for line in open(f_name, "r", encoding="utf-8"):
            label, line_ = line.split("\t")
            labels.append(label)
            lines.append(line_)
            for w in line_.split():
                self.ws[w] = 0
        #pprint(self.ws)

        # train the model: update the weights
        for iter_ in range(iter_times_training):
            for label, line in zip(labels, lines):
                #print(label,line)
                phi_dict=self.phi(line)
                val=abs(self.sum_prod(phi_dict))
                if val<=10:
                    self.update_weight(label, phi_dict)

        # generate string showing the predicted value of line
        self.str = ""
        for line in lines:
            label = self.predict_one_line(line)
            predicted_line = "{label}\t{sentence}\n".format(
                label=label, sentence=line)
            self.str += predicted_line

    def predict_one_line(self, line):
        phi_dict = self.phi(line)
        sum_product= self.sum_prod(phi_dict)
        output=(math.e**sum_product)/float((1+math.e)**sum_product)
        print(sum_product,output)
        return sum_product

    def __str__(self):
        return self.str

    def update_weight(self, label, phi_dict, c=0.1,alpha=0.01):
        #phi_dict = self.phi(line)
        # print(phi_dict)
        for w in phi_dict:
            if abs(self.ws[w])<c:
                self.ws[w]=0
            else:
                self.ws[w]-=sign(self.ws[w])*c
        w__phi_x = self.sum_prod(phi_dict)
        coeff_phi = alpha * self.grad_P_coeff_phi(label, phi_dict)
        for w in phi_dict:
            self.ws[w] += coeff_phi * phi_dict[w]
            #print("coeff phi:\t{coeff}\n{weight}".format(coeff=coeff_phi,weight=self.ws[w]))
            #input()
            
            

    def grad_P_coeff_phi(self, label, phi_dict):
        #print(label,phi_dict,sep="\n")
        label=int(label)
        coeff = 0
        sum_prod = self.sum_prod(phi_dict)
        #print(sum_prod)
        if label == -1:
            coeff = -math.e**(sum_prod) / ((float(1 + math.e**(sum_prod)))**2)
        elif label == 1:
            coeff = math.e**(sum_prod) / ((float(1 + math.e**(sum_prod)))**2)
        #print(coeff)
        #input()
        return coeff

    def phi(self, line):
        phi_dict = defaultdict(int)
        for w in line.split():
            phi_dict[w] += 1
        return phi_dict

    def sum_prod(self, phi_dict):
        #print(phi_dict)
        output = 0
        for w in phi_dict:
            output += self.ws[w] * phi_dict[w]
        #input(output)
        return output
    
    def predict(self, f_name):
        for line in open(f_name,"r",encoding="utf-8"):
            predicated_label=self.predict_one_line(line)
            print("{predict}\t{sentence}".format(predict=predicated_label,sentence=line))

# In[117]:

training_file_path="../../data/titles-en-train.labeled"


# In[118]:

if __name__=="__main__":
    sgd=SGD(training_file_path)


# In[119]:

print(sgd)


# In[59]:

math.e**0


# In[ ]:



