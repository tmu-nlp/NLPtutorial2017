
# coding: utf-8

# In[1]:

import os,sys,subprocess
from collections import defaultdict


# In[3]:

class Perceptron():
    def __init__(self,input_f=None):
        """if input_f is given, train the perception based on this file
        if input_f not given, use self.load_model() to read the model
        model file line shoud be in the format of:
            class\tline"""
        
        self.w=defaultdict(lambda:0.0)
        self.phi=defaultdict(lambda:0)
        lines=[]
        ys=[]
        
        if input_f:
            with open(input_f,"r",encoding="utf-8") as f:
                for line_ in f:
                    class_,line=line_.split("\t")
                    for i in line.split():
                        self.phi[i]+=1
                    lines.append(line)
                    ys.append(int(class_))
            for i in self.phi:
                self.w[i]=0
            
            # train the data 10 iters on all the lines
            self.model_flag=True
            self.train_model(10,lines,ys)
        else:
            self.model_flag=False
    
    def gen_phi(self,line):
        output=defaultdict(lambda:0)
        for i in line.split():
            output[i]+=1
        return output
    
    def train_model(self,n_iter,lines,ys):
        """train model for several more iterations if the model is loaded"""
        if self.model_flag:
            for i in range(n_iter):
                for line,y in zip(lines,ys):
                    phi=self.gen_phi(line)
                    y_=self.predict_1(phi)
                    if y_!=y:
                        self.update_weights(phi,y)
        else:
            print("please either load model or init a modle with training corpus.")
    def update_weights(self,phi,y):
        for i,value in phi.items():
            self.w[i]+=value*y
    def save_model(self,f_name):
        with open(f_name, "w",encoding="utf-8") as f:
            for i in self.w:
                f.write("{word}\t{value}\n".format(word=i,value=self.w[i]))
        with open(f_name, "r",encoding="utf-8") as f:
            for i in range(5):
                print(f.readline())
    
    def load_model(self,f_name):
        """load the model based on the given file"""
        if not self.model_flag:
            with open(f_name,"r",encoding="utf-8") as f:
                for line in f:
                    line=line[:-1]
                    word,value = line.split("\t")
                    self.w[word]=float(value)
        else:
            overwrite_flag=input("model already exist, overwrite? (input y or n)")
            if overwrite_flag=="y":
                self.model_flag=False
                self.load_model(f_name)
            elif overwrite_flag=="n":
                pass
            else:
                print("please input y or n")
                self.load_model(f_name)
    
    def predict_1(self,phi):
        """predict using self model"""
        output=0
        for i in phi:
            output+=self.w[i]*phi[i]
        if output>=0:
            _=1
        else:
            _=-1
        return _
    def predict(self, f_name):
        with open(f_name,"r",encoding="utf-8") as f:
            with open(f_name+"_classification","w",encoding="utf-8") as fo:
                for line in f:
                    class_=self.predict_line(line)
                    fo.write(str(class_)+"\t"+line)
    
    def predict_line(self,line):
        phi=self.gen_phi(line)
        print(self.predict_1(phi))

                



if __name__=="__main__":
    percep=Perceptron("./../../data/titles-en-train.labeled")
    percep.predict("./../../data/titles-en-test.word")
    input()
