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
#from spacy import displacy


def Get_RootTokens(doc):
    a = [token for token in doc if(str(token.dep_)=="ROOT")]
    return a
def Get_AttrOfRoot(doc,root):
     a = [token for token in doc if(root.is_ancestor(token) and (str(token.dep_) in ADJECTIVES))]
     return a
def Get_SubjOfRoot(doc,root):
    a = [token for token in doc if(root.is_ancestor(token) and((str(token.dep_) in SUBJECTS) or (str(token.dep_) in OBJECTS)))]
    return a
def Get_NegOfSubj(doc,subj,root):
    a = []
    for token in doc:
        if(root.is_ancestor(token) and (str(token.dep_) in NEGATIONS) and (root.is_ancestor(subj) and root.is_ancestor(subj))):
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
                if(neg.head in GetHeads(doc,token.head)):
                    b.append(neg)
            a.append(b)
    return a


nlp = spacy.load('en_core_web_sm')

while(True):
    print('--------------------------')
    print('--------------------------')
    doc2 = nlp(input('Enter String: '))
    
    for token in doc2:
        for token2 in doc2:
          print(token.text,token2.text,token.dep_,token.is_ancestor(token2),token2.dep_,token2.is_ancestor(token))
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
             attrsS = [k for k in Get_AttrOfRoot(doc2,subj)]
             NegW = [k for k in Get_NegOfAttr(doc2,attrsR,subj,root)]
             print(subj,':',attrsR,';',attrsS,';;',NegW)
   # displacy.serve(doc2, style='doc')

"""DONE
1. Subject extractions
2.  Finding attributes and complementing words like 'not'
3. Model to evaluate words
---
TO DO
---
1. Find the negative words association to attributes.
2. What do to with 'very', 'little' etc.
3.  Import model to evaluate final number- call model.load() and use compute function
4.  
"""
