# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 16:51:56 2018

@author: NikhilR

Text: The original word text.
Lemma: The base form of the word.
POS: The simple part-of-speech tag.
Tag: The detailed part-of-speech tag.
Dep: Syntactic dependency, i.e. the relation between tokens.
Shape: The word shape – capitalisation, punctuation, digits.
is alpha: Is the token an alpha character?
is stop: Is the token part of a stop list, i.e. 
the most common words of the language?
"""
SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
OBJECTS = ["dobj", "dative", "attr", "oprd"]
ADJECTIVES = ["acomp", "advcl", "advmod", "amod", "appos", "nn", "nmod", "ccomp", "complm",
              "hmod", "infmod", "xcomp", "rcmod", "poss"," possessive"]
COMPOUNDS = ["compound"]
PREPOSITIONS = ["prep"]
NEGATIONS = ["no", "not", "n't", "never", "none","neg"]
import spacy
import tensorflow as tf
import tensorflow.keras as keras
#from spacy import displacy
imdb = keras.datasets.imdb
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3
def encode_review(string):
    w, h = 256, 1
    Matrix = [[0 for x in range(w)] for y in range(h)] 
    Matrix[0][0] = 1
    i = 1
    for x in string.split(' '):
       if(word_index.get(x)!=None):
           Matrix[0][i]=word_index[x]
       else:
           Matrix[0][i]= (2)
       i+=1
    Matrix[0] = keras.preprocessing.sequence.pad_sequences([Matrix[0]],value = word_index["<PAD>"],padding='post',maxlen=256)
    m = Matrix[0]
    #rez = [[m[j][i] for j in range(len(m))] for i in range(1)]     
    return(m)

def Get_RootTokens(doc):
    a = [token for token in doc if(str(token.dep_)=="ROOT")]
    return a
def Get_AttrOfRoot(doc,root):
     a = [token for token in doc if(root == (token.head) and (str(token.dep_) in ADJECTIVES))]
     return a
def Get_SubjOfRoot(doc,root):
    a = [token for token in doc if(root.is_ancestor(token) and((str(token.dep_) in SUBJECTS) or (str(token.dep_) in OBJECTS)))]
    return a
def Get_NegOfSubj(doc,subj,root):
    a = []
    for token in doc:
        if(root.is_ancestor(token) and (str(token.dep_) in NEGATIONS) and root.is_ancestor(subj) and (subj.head==token.head)):
            a.append(token)
    return a
def GetHeads(doc,token):
    a=[]
    while(True):
        a.append(token)
        if(token.dep_=="ROOT" or token.dep_=="VERB" or token.dep_=="NOUN"):
            break
        else:
            token = token.head
    return a
def Get_NegOfAttr(doc,attrs,subj,root):
    a = []
    for token in doc:
        if(token in attrs):
            negw = Get_NegOfSubj(doc,token,root)
            b=[]
            for neg in negw:
                #if(neg.head==token.head.head or neg.head == token.head):
                if(neg.head in GetHeads(doc,token.head) and not b.__contains__(neg)):
                    b.append(neg)        
            if(not a.__contains__(b)):    
                a.append(b)
    return a


nlp = spacy.load('en_core_web_sm')
model = keras.models.load_model('Model.bin');
while(True):
    print('--------------------------')
    print('--------------------------')
    doc2 = nlp(input('Enter String: '))
    
    #"""for token in doc2:
     #   for token2 in doc2:
      #    print(token.text,token2.text,token.dep_,token.is_ancestor(token2),token2.dep_,token2.is_ancestor(token))
    #"""
    print('--------------------------')
    print('--------------------------')
    if(doc2.text=='break'):
        break
    Roots = Get_RootTokens(doc2)
    for root in Roots:
        subjs = Get_SubjOfRoot(doc2,root)
        print(root,'-')
        for subj in subjs:
             attrsR = [k for k in Get_AttrOfRoot(doc2,subj.head)]
             attrsS = [ float(model.predict(encode_review(k.string))) for k in Get_AttrOfRoot(doc2,subj.head)]
             NegW = [k for k in Get_NegOfAttr(doc2,attrsR,subj,root)]
             print(subj,':',attrsR,';',attrsS,';;',NegW)
                
    # displacy.serve(doc2, style='doc')

"""DONE
1. Subject extractions
2. Finding attributes and complementing words like 'not'
3. Model to evaluate words
4. Import model to evaluate final number- call model.load() and use compute function
5. Find the negative words association to attributes.
---
TO DO
---
1. What do to with 'very', 'little' etc.
2. Parse data as JSON/XML to C# for further analysis.  
3. Adding multiple numbers/adjectives??
"""
