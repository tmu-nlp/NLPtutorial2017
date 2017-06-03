
# coding: utf-8

# In[16]:

import sys,pprint,math
from collections import defaultdict


# In[17]:

class pos_HMM(object):
    def __init__(self,f_name):
        
        self.w_pos_freq=defaultdict(lambda:0)
        self.pos_2gram_freq={   # contains both 1 and 2 gram frequency
            1:defaultdict(lambda:0),
            2:defaultdict(lambda:0)
        }
        
        
        for line in open(f_name, "r", encoding="UTF-8"):  
            # count all the w_pos pair freq, and 1 and 2 gram freq
            line="<s>_<s> "+ line.replace("\n","") + " </s>_</s>"
            tags=[]
            tags[:]=[]
            for p in line.split():
                w,tag =p.split("_")
                self.w_pos_freq[(w,tag)]+=1
                self.pos_2gram_freq[1][tag]+=1
                tags.append(tag)
            for bigram in zip(tags, tags[1:]):
                self.pos_2gram_freq[2][bigram]+=1
        
    def calc_transmission_P(self, tag, tag_prev):      
        # calculate the transmission P based on the 1 and 2 gram freq dict, no smoothing is needed
        P=self.pos_2gram_freq[2][(tag_prev,tag)]/float(self.pos_2gram_freq[1][tag_prev])
        return P   #  output can be 0 if check unseen sequence
    
    def calc_emission_P(self, tag, w, lambda_=0.95, vocab_size=1000000):
        # calculate the emission P based on the w_pos freq dictionary, smoothing is needed
        P=lambda_*self.w_pos_freq[(w,tag)]/float(self.pos_2gram_freq[1][tag])+(1-lambda_)/vocab_size
        return P
    
    def gen_pos_tags(self, line_str):
        # generate pos tag list using viterbi matrix and viterbi algo
        
        # generate viterbi matrix: best_score (init with 1000000) and best_prev_node (init with empty value)
        line_str="<s> "+line_str+" </s>"
        best_score=dict()
        best_prev_node=dict()
        for ind,w in enumerate(line_str.split()):
            best_score[(ind,w)]=dict()
            best_prev_node[(ind,w)]=dict()
            if w=="<s>":
                best_score[(ind,w)][(ind,"<s>")]=0
                best_prev_node[(ind,w)][(ind,"<s>")]=None
            elif w=="</s>":
                best_score[(ind,w)][(ind,"</s>")]=1000000.0
                best_prev_node[(ind,w)][(ind,"</s>")]=(0,"")
            else:
                tags=[t for t in self.pos_2gram_freq[1] if t != "<s>" and t != "</s>"]
                for t in tags:
                    best_score[(ind,w)][(ind,t)]=1000000.0
                    best_prev_node[(ind,w)][(ind,t)]=(0,"")
        #pprint.pprint(best_score)
        #pprint.pprint(best_prev_node)
        
        # forward step: calculate best socre and prev node on each node
        line_list=line_str.split()
        for ind, w in enumerate(line_list):
            if ind==0:
                pass
            
            else:
                w_prev=line_list[ind-1]
                for ind_, tag in best_score[ind,w]:
                    for ind_prev,tag_prev in best_score[ind-1,w_prev]:
                        trans_P=self.calc_transmission_P(tag,tag_prev)
                        emis_P=self.calc_emission_P(tag,w)
                        if trans_P!=0:
                            score=-math.log(trans_P)+ -math.log(emis_P)
                        else:
                            score=100000.0
                        score+=best_score[(ind_prev,w_prev)][(ind_prev,tag_prev)]
                        if score<best_score[(ind,w)][(ind,tag)]:
                            best_score[(ind,w)][(ind,tag)]=score
                            best_prev_node[(ind,w)][(ind,tag)]=(ind_prev,tag_prev)
        #pprint.pprint(best_score)
        #pprint.pprint(best_prev_node)
        
        # backword step:
        ind=len(line_list)-1
        w=line_list[ind]
        tag="</s>"
        #print(ind,w,tag)
        #input()
        
        pos_tags=[]
        while best_prev_node[ind,w][ind,tag]:
            ind,tag=best_prev_node[ind,w][ind,tag]
            pos_tags.append(tag)
            w=line_list[ind]
            
            
        pos_tags.reverse()
        
        
        return " ".join(pos_tags[1:])
    def gen_POS_tags_f(self, f_name):
        # call the self.gen_POS_tag function line by line given the input file
        output=[]
        for line in open(f_name,"r",encoding="UTF-8"):
            output.append(self.gen_pos_tags(line)+"\n")
        return output
    


# In[18]:

temp_model=pos_HMM("./wiki-en-train.norm_pos")


# In[21]:

file_to_tag=["./wiki-en-train.norm","./wiki-en-test.norm"]
for f in file_to_tag:
    with open("my-answer_"+f[2:],"w",encoding="UTF-8") as output_f:
        output_f.writelines(temp_model.gen_POS_tags_f(f))


# In[ ]:

temp_model.pos_2gram_freq[1]


# In[ ]:

print(temp_model.calc_transmission_P("CC","CC"))


# In[ ]:

print(temp_model.calc_emission_P("NN","rea"))


# In[ ]:

pprint.pprint(temp_model.w_pos_freq)


# In[ ]:



