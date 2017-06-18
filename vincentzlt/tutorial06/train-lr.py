import sys,math,random
from collections import defaultdict
from pprint import pprint

class LR():
    def __init__(self, f_name):
        self.weights=defaultdict(float)
        #phi_dict_all=defaultdict(int)

        self.training_data=[]

        # read f_name into lines and labels
        with open(f_name,"r",encoding="utf-8") as f:
            for line in f:
                label,line=line.split("\t")
                self.training_data.append((label,line))
        
        # 
    # compute the logistic prob of a sentence over phi_dict
    def logistic(self,line):
        e=math.e
        ws=self.weights
        phi_dict=self.phi(line)
        sum_prod_w_phi=self.sum_prod(phi_dict)
        p=(e**sum_prod_w_phi)/(1+e**sum_prod_w_phi)
        return p
    
    # predict one sentence
    def predict_one(self,line):
        p=self.logistic(line)
        if p>=0.5:
            return 1
        else:
            return -1
    
    def predict_all(self,f_name):
        with open(f_name,"r",encoding="utf-8") as f:
            for line in f:
                print(self.predict_one,line,sep='\t')
    # compute the sum of prod over self.weight dict and phi_dict
    def sum_prod(self, phi_dict):
        sum=0
        ws=self.weights
        for w in phi_dict:
            sum+=ws[w]*phi_dict[w]
        return sum
    # train the model with iter_times(100) and learning_rate alpha(0,001)
    def train(self, iter_times=100, alpha=0.001, margin=10,c=1, testdata=None):
        accuracies=[]
        for ind,iter_ in enumerate(range(iter_times)):
            for label,line in self.training_data:
                #print(label,line,sep='\t')
                phi_dict=self.phi(line)

                exp_e=self.sum_prod(phi_dict)
                if exp_e*int(label) < margin:
                    update_phi_dict=self.grad_logistic(label,phi_dict,exp_e)
                    for w in update_phi_dict:
                        if abs(self.weights[w])<c:
                            self.weights[w]=0
                        else:
                            self.weights[w]-=sign(self.weights[w])*c

                        self.weights[w]+=alpha*update_phi_dict[w]
            
            if ind%10==0:
                testdata=self.sample(self.training_data,0.1)
                accuracy=self.test(testdata)
                print("epich {ind}:\t{accuracy}".format(ind=ind,accuracy=accuracy))
                accuracies.append(accuracy)
                if sum(accuracies[-3:])/3>0.97:
                    
                    break
    def sample(self,data,sample_rate=0.1):
        
        for i in data:
            if random.random()<sample_rate:
                yield i

    def test(self,test_data):
        result=[]
        for label,sent in test_data:
            if self.predict_one(sent)==int(label):
                result.append(1)
            else:
                result.append(0)
        if len(result)>0:  
            return sum(result)/len(result)
        else:
            input()
    
    def test_f(self,test_f):
        testing_data=[]
        with open(test_f,"r",encoding="utf-8") as f:
            for line in f:
                label, sent = line.split("\t")
                testing_data.append((label,line))
        print(self.test(testing_data))
    # compute the grad descent value of logistic unction based on the label and sentence
    def grad_logistic(self, label, phi_dict,exp_e=None):
        e=math.e
        label=int(label)
        if not exp_e:
            exp_e=self.sum_prod(phi_dict)
        if label==-1:
            coeff_phi=-(e**exp_e)/((1+e**exp_e)**2)
        if label==1:
            coeff_phi=(e**exp_e)/((1+e**exp_e)**2)
        for w in phi_dict:
            phi_dict[w]*=coeff_phi
        return phi_dict
    def phi(self,line):
        phi_dict=defaultdict(int)
        for w in line.split():
            phi_dict[w]+=1
        return phi_dict
    
                
def sign(number):
    if number>=0:
        return 1
    else:
        return -1

if __name__=="__main__":
    lr=LR("../../data/titles-en-train.labeled")
    lr.train(iter_times=100)
    lr.test_f("../../data/titles-en-test.labeled")
    input()
                


