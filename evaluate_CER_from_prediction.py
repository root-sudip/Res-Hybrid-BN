from __future__ import print_function
import h5py,math
import numpy as np
import os,sys
from PIL import Image,ImageOps
import editdistance as ed

def evaluate_word_accuracy(target,prediction):
    #both strings in bangla format separated words by space(*)
    target_words=target.split("*")
    predicted_words=prediction.split("*")
    nbwords=len(target_words)
    accuracy_vector = np.zeros([nbwords])
    errors=0
    for w in range(nbwords):
        try:
            if(predicted_words[w]==target_words[w]):
                accuracy_vector[w]=1
            else:
                errors+=1
        except:
            pass
    av_string=""
    for a in accuracy_vector:
        av_string=av_string+" "+str(a)
    return errors,nbwords,av_string

def evaluate_CER_from_prediction(predictionfile):
    f=open(predictionfile)
    line=f.readline()
    total_chars=0
    total_ce=0
    total_words=0
    total_we=0
    while line:
        info=line.strip("\n").split(",")
        true=info[1].strip("*")
        predicted=info[2].strip("*")
        print("Comparing %s with %s"%(true,predicted))
        ce=ed.eval(true,predicted)
        nbchars=len(true)
        total_chars+=nbchars
        total_ce+=ce
        we,nbwords,_=evaluate_word_accuracy(true,predicted)
        total_we+=we
        total_words+=nbwords
        line=f.readline()
    cer=total_ce/float(total_chars)
    wer=total_we/float(total_words)
    print("Total characters %d total ce %d CER %f Character accuracy %f "%(total_chars,total_ce,cer,(1-cer)))
    print("Total words %d total we %d WER %f Word accuracy %f " % (total_words, total_we, wer, (1-wer)))

evaluate_CER_from_prediction("Predicted.txt")
