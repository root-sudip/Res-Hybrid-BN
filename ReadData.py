from __future__ import print_function
import h5py,math
import numpy as np
import os,sys
from PIL import Image,ImageOps
import editdistance as ed

def rescale_image(image_mat,nw,nh,normalize=True,invert=True):
    img=Image.fromarray(image_mat.astype('uint8')).convert('L')
    w,h=img.size[0],img.size[1]
    ar = w / float(h)
    new_w = int(ar*nh)
    if(new_w>nw):
        new_w=nw
    canvas = Image.new('L', (nw, nh),color='white')
    rescaled=img.resize((new_w,nh))
    canvas.paste(rescaled)
    if(invert):
        canvas=ImageOps.invert(canvas)
    new_image_mat=np.reshape(canvas.getdata(),[nh,nw])
    if(normalize):
        new_image_mat=new_image_mat/float(255.0)
    return new_image_mat,new_w,h


def batch_rescale_image(imagemats,nw,nh):
    rescale_image_mats=[]
    old_hs=[]
    old_ws=[]
    for mat in imagemats:
        new_mat,old_w,old_h=rescale_image(mat,nw,nh,normalize=False)
        rescale_image_mats.append(new_mat)
        old_hs.append(old_h)
        old_ws.append(old_w)
    return rescale_image_mats,old_ws,old_hs

def test_image_rescaling(imfile,nw,nh):
    img=Image.open(imfile).convert('L')
    r,c=img.size[1],img.size[0]
    print(r,c)
    img=np.reshape(img.getdata(),(r,c))
    rescaled=rescale_image(img,nw,nh,save=False)
    ims=Image.fromarray(rescaled.astype('uint8')).save('rescaled.png')
    print(rescaled.shape)


def readMainH5(h5file,readamount,write_sequences=False,binarize=False):
    #Reads a standard format h5 file and collects features, targets and sequence_lengths
    #maxw and maxh are maximum width(cols) and maximum height(rows) of all samples train+test+val
    f=h5py.File(h5file)
    samplenames=list(f.keys())
    total=len(samplenames)
    all_x=[]
    all_y=[]
    all_sampleid=[]
    seq_lengths=[]
    total_target_chars=0
    if(write_sequences):
        seq_file = open("Sequence_lengths", "w")
        seq_file.write(h5file+"\n")
    for t in range(total):
            completed=(t/float(total))*100
            if (completed >= readamount):
                break
            sample=f.get(samplenames[t])
            bangla_target=sample.attrs['Bangla_Target']
            target = sample.attrs["Reorder_Target"]
            target_chars=target.split()
            flag=False
            target_length = len(target_chars)
            total_target_chars+=target_length
            for ch in target_chars:
                if(ch!='*'):
                    flag=True
                    break
            if(not flag):
                print("\tBad sample ",sample.name)
                continue

            sampleid=str(sample.name)
            all_sampleid.append(sampleid)
            features=np.asarray(sample.get("Image")) # H x W
            h,w=features.shape[0],features.shape[1]
            if(binarize):
                features=convert_to_binary(features)
            all_x.append(features) # N x W x H
            all_y.append(target)
            seq_lengths.append(w)
            if(write_sequences):
                seq_file.write(sampleid+","+bangla_target+","+str(w)+","+str(h)+","+str(target_length) + "\n")


    print("Reading ",h5file," complete")
    if(write_sequences):
        seq_file.close()
    return all_x,all_y,seq_lengths,all_sampleid,total_target_chars

#Second
def findDistinctCharacters(targets):
    '''
    Reads all targets (targets) and splits them to extract individual characters
    Creates an array of character-integer map (char_int)
    Finds the maximum target length
    Finds number of distinct characters (nbclasses)
    :param targets:
    :return char_int,max_target_length,nbclasses:
    '''
    total=len(targets)
    max_target_length=0
    char_int=[]
    all_chars=[]
    total_transcription_length=0 #Total number of characters
    for t in range(total):
        this_target=targets[t]
        chars=this_target.split()
        target_length=len(chars)
        total_transcription_length=total_transcription_length+target_length
        if(target_length>max_target_length):
            max_target_length=target_length
        for ch in chars:
            all_chars.append(ch)

    charset = list(set(all_chars))
    nbclasses = len(charset)
    print("Character Set processed for ", total, " data")
    return charset, max_target_length, nbclasses, total_transcription_length

def pad_x_single(x,maxdim):
    rows=maxdim[0]
    cols=maxdim[1]
    padded_x=np.zeros([rows,cols,1])
    for r in range(len(x)):
        for c in range(len(x[r])):
            padded_x[r][c][0]=x[r][c]
    return padded_x

#Third
def pad_x(x,maxdim):
    total=len(x)
    rows=maxdim[0]
    cols=maxdim[1]
    padded_x=np.zeros([total,rows,cols,1])
    for t in range(total):
        for r in range(len(x[t])):
            for c in range(len(x[t][r])):
                padded_x[t][r][c][0]=x[t][r][c]
    return padded_x

#Call inside Training Module
def make_sparse_y(targets,char_int,max_target_length):
    total = len(targets)
    indices=[]
    values=[]
    shape=[total,max_target_length]
    for t in range(total):
        chars=targets[t].split()
        for c_pos in range(len(chars)):
            sparse_pos=[t,c_pos]
            sparse_val=char_int.index(chars[c_pos])
            indices.append(sparse_pos)
            values.append(sparse_val)
    return [indices,values,shape]

def find_max_dims(hdfin):
    f=h5py.File(hdfin)
    meta=open('Image_info.txt','w')
    nh=40
    keys=list(f.keys())
    total=len(keys)
    maxw=0
    maxh=0
    for t in range(total):
        sample=f.get(keys[t])
        data=np.asarray(sample.get('Image'))
        w=data.shape[1]
        h=data.shape[0]
        samplename=sample.name
        ar=w/float(h)
        nw=ar*nh
        meta.write("%s,%d,%d,%d,%d\n"%(samplename,w,h,nw,nh))
        print("Reading %s"%samplename)
        if(nw>maxw):
            maxw=nw
        if(nh>maxh):
            maxh=nh
    meta.write("MaxW,%d,MaxH,%d\n"%(maxw,maxh))
    meta.close()
    return maxw,maxh


#Adjust Sequence lengths after CNN and Pooling
def adjustSequencelengths(seqlen,reduction,limit):
    total=len(seqlen)
    for s in range(total):
        seqlen[s]=min(limit,math.ceil(seqlen[s]/float(reduction)))
    return seqlen
#Main
def load_data(trainh5,testh5,batchsize,readamount,generate_char_table):
    train_x, train_y, train_seq_lengths,train_sampleids,_=readMainH5(trainh5,readamount,write_sequences=True,binarize=False)
    test_x,test_y,test_seq_lengths,test_sampleids,_=readMainH5(testh5,readamount, binarize=False)

    sampleids=[train_sampleids,test_sampleids]

    train_charset, train_max_target_length, train_nbclasses,train_transcription_length=findDistinctCharacters(train_y)
    test_charset, test_max_target_length, test_nbclasses, test_trainscription_length = findDistinctCharacters(test_y)
    print("Train Char Set ", train_nbclasses, " Test Character set ", test_nbclasses)

    if (train_nbclasses < test_nbclasses):
        print("Warning ! Test set have more characters")

    train_charset.extend(test_charset)

    char_int = []
    if (generate_char_table):
        charset = list(set(train_charset))  # A combined Character set is created from Train and test Character set
        charset.sort()
        charset.insert(0, "PD")
        charset.append("BLANK")
        nb_classes = len(charset)  # For Blank

        for ch in charset:
            char_int.append(ch)

        ci = open("Character_Integer", "w")
        for ch in char_int:
            ci.write(ch + "\n")
        ci.close()
        print("Character Table Generated and Written")
    else:
        ci = open("Character_Integer")
        line = ci.readline()
        while line:
            char = line.strip("\n")
            char_int.append(char)
            line = ci.readline()
        nb_classes = len(char_int)
        print("Character Table Loaded from Generated File")

    max_target_length=max(train_max_target_length,test_max_target_length)
    config=open("Config.txt",'a')
    config.write(str(max_target_length)+"\n")
    config.close()

    nbtrain=len(train_y)
    nbtest=len(test_y)

    y_train=[]
    y_test=[]

    batches=int(np.ceil(nbtrain/float(batchsize)))
    start=0
    for b in range(batches):
        end=min(nbtrain,start+batchsize)
        sparse_target=make_sparse_y(train_y[start:end],char_int,max_target_length)
        y_train.append(sparse_target)
        start=end

    batches = int(np.ceil(nbtest / float(batchsize)))
    start = 0
    for b in range(batches):
        end = min(nbtest, start + batchsize)
        sparse_target = make_sparse_y(test_y[start:end], char_int, max_target_length)
        y_test.append(sparse_target)
        start = end
    transcription_length=[train_transcription_length,test_trainscription_length]

    return [train_x,test_x],nb_classes,[train_seq_lengths,test_seq_lengths],[y_train,y_test],max_target_length,char_int,transcription_length,sampleids

#Convert integer representation of string to unicode representation
def int_to_bangla(intarray,char_int_file,dbfile):
    '''
    Takes an array of integers (each representing a character as given in char_int_file
    dbfile contains global mapping
    :param intarray:
    :param char_int_file:
    :param dbfile:
    :return:unicode string,mapped character string
    '''
    char_int=[]
    f=open(char_int_file)
    line=f.readline()
    while line:
        info=line.strip("\n")
        char_int.append(info)
        line=f.readline()
    f.close()

    chars=[]
    for i in intarray:
        chars.append(char_int[i])

    banglastring=""
    for ch in chars:
        f=open(dbfile)
        line=f.readline()
        while line:
            info=line.strip("\n").split(",")
            if(info[2]==ch):
                banglastring=banglastring+info[1]+" "
            line=f.readline()
        f.close()
    return banglastring,chars

def find_unicode_info(char,dbfile):
    #returns type and actual unicode position of a character
    f=open(dbfile)
    line=f.readline()
    type="v"
    pos="#"
    while line:
        info=line.strip("\n").split(",")
        if(len(info)>5):
            line=f.readline()
        else:
            if(char==info[1]):#Found it in DB
                type=info[0]
                if(type=="m"):#its a modifier
                    pos=info[-1]
                break
            line=f.readline()
    f.close()
    return [type,pos]

def find_character_frequency(h5file,statfile):
    f=h5py.File(h5file)
    keys=list(f.keys())
    total=len(keys)
    targets=[]
    for t in range(total):
        sample=f.get(keys[t])
        target=sample.attrs["Custom_Target"]
        targets.append(target)
        print("Reading ",sample.attrs["SampleID"]," Target ",target)
    f.close()

    all_chars=[]
    for tg in targets:
        characters=tg.split()
        for ch in characters:
            all_chars.append(ch)

    unique_chars=list(set(all_chars))
    unique_chars.sort()
    nb_classes=len(unique_chars)
    hist=np.zeros([nb_classes])

    for tg in targets:
        characters=tg.split()
        for ch in characters:
            ind=unique_chars.index(ch)
            hist[ind]=hist[ind]+1

    print(unique_chars)
    print(hist)

    dict = open("Dict/bengalichardb.txt")
    line = dict.readline()
    unicode_custom=[]
    while line:
        info = line.strip("\n").split(",")
        if(len(info)==4) or (len(info)==5):
            unicode_map=[info[2],info[3]]
            unicode_custom.append(unicode_map)
        line=dict.readline()
    dict.close()

    outfile=open(statfile,"w")

    for i in range(nb_classes):
        for j in range(len(unicode_custom)):
            if(str(unique_chars[i])==unicode_custom[j][1]):
                unicode_value=str(unicode_custom[j][0])
                break
        outfile.write(unicode_value+","+str(unique_chars[i])+","+str(hist[i])+"\n")
    outfile.close()

def reset_unicode_order(unicodestring,dbfile):
    #Takes unicodestring seperated by space
    #returns properly ordered unicodestring
    unicodearray=unicodestring.split()
    unicodearray=[ch.decode("utf-8").encode("unicode-escape") for ch in unicodearray]
    nbchars=len(unicodearray)
    i=0
    while (i<nbchars-2):
        [type, pos]=find_unicode_info(unicodearray[i],dbfile)
        if(type=="m"):# May need swap
            if(pos=="p"):#swap
                temp=unicodearray[i]
                unicodearray[i]=unicodearray[i+1]
                unicodearray[i+1]=temp
                i=i+1
        i=i+1
    reorder_string=""
    for u in unicodearray:
        reorder_string=reorder_string+u.encode("utf-8").decode("unicode-escape")
    return reorder_string

def gather_offline_info(dir):
    widths=[]
    heights=[]
    for root,sd,files in os.walk(dir):
        for fname in files:
            print("Reading ",fname)
            if(fname[-4:]=="jpeg"):
                absfname=os.path.join(root,fname)
                im=Image.open(absfname)
                print(im.size) #cols, rows
                widths.append(im.size[0])
                heights.append(im.size[1])
    max_width=max(widths)
    max_height=max(heights)
    print("Max W=",max_width," Max H=",max_height)

def convert_to_binary(image):
    #image shoud have dim W x H
    r=len(image)
    c=len(image[0])
    for i in range(r):
        for j in range(c):
            if(image[i][j]==255):
                image[i][j]=1
            else:
                image[i][j]=0
    return image

def load_compound_map(dbfile):
    f=open(dbfile)
    line=f.readline()
    compound_dict={}
    compound_dict['*']="S"
    while line:
        info=line.strip("\n").split(",")
        comp=info[-1]
        unicd=info[0]
        compound_dict[comp]=unicd
        line=f.readline()
    print("Compound map loaded")
    return compound_dict

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

def load_character_integer():
    f=open("Character_Integer")
    line=f.readline()
    char_int=[]
    while line:
        info=line.strip("\n")
        char_int.append(info)
        line=f.readline()
    print("Character integer loaded")
    return char_int

def load_test_data(testhdf,batchsize):
    all_x, all_y, seq_lengths, all_sampleid,transcription_len=readMainH5(testhdf,100,write_sequences=True)
    nbtest=len(all_y)
    batches = int(np.ceil(nbtest / float(batchsize)))
    char_int=load_character_integer()
    start = 0
    y_test=[]
    for b in range(batches):
        end = min(nbtest, start + batchsize)
        sparse_target = make_sparse_y(all_y[start:end], char_int, 78)
        y_test.append(sparse_target)
        start = end
    return all_x,y_test,seq_lengths,all_sampleid,transcription_len

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
    print("Total characters %d total ce %d CER %f"%(total_chars,total_ce,cer))
    print("Total words %d total we %d WER %f" % (total_words, total_we, wer))
