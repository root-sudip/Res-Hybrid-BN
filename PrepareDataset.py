from __future__ import print_function
import os,shutil,csv,sys
import h5py
from PIL import Image
import numpy as np
from UnicodeProcess import *
from random import shuffle

dbfile="Dict/CompositeAndSingleCharacters.txt"
compounddbfile="Dict/BanglaCompositeMap.txt"

def split_train_test(dir,source,destination):
    all_index=[]
    for root,sd,files in os.walk(dir):
        for fname in files:
            print(fname)
            if(fname[-3:]=="txt"):
                index=fname.split("_")[1].split(".")[0]
                all_index.append(index)
    print("File list ready")
    total=0
    for root,sd,files in os.walk(source):
        for fname in files:
            if(fname[-3:]=="tif"):
                index = fname.split("_")[1].split(".")[0]
                for i in range(len(all_index)):
                    if(all_index[i]==index):
                        absfilename = os.path.join(root, fname)
                        shutil.move(absfilename, destination)
                        total = total + 1
                        break
                print("Moved ",fname)
    print('Total ',total,' files moved')

def group_image_with_groundtruth(dir):
    images=[]
    groundtruth=[]
    for root,sd,files in os.walk(dir):
        for fname in files:
            if(fname[-3:]=="tif"):
                images.append(os.path.join(root,fname))
            elif(fname[-3:]=="txt"):
                groundtruth.append(os.path.join(root,fname))
    print('File list ready')

    data=[]

    for gt in groundtruth:
        f=open(gt)
        line_no = 1
        line=f.readline()
        while line:
            target=line.strip("\n")
            fname=gt.split("/")[-1][:-4]
            image_name=fname+"_"+"line_"+str(line_no)+".tif"
            data.append([image_name,target])
            line_no+=1
            line=f.readline()
        f.close()

    outfile=open(dir+"/found.txt","w")
    nf=open(dir+"/notfound.txt","w")
    for d in data:
        imname=d[0]
        found=False
        for i in images:
            fname=i.split("/")[-1]
            if(fname==imname):
                outfile.write(i+","+d[1]+"\n")
                found=True
                break
        if(found==False):
            nf.write(imname+","+d[1]+"\n")
    outfile.close()
    nf.close()

def read_D2_format(dir):
    images = []
    groundtruth = []
    for root, sd, files in os.walk(dir):
        for fname in files:
            if (fname[-3:] == "csv"):
                groundtruth.append([root,os.path.join(root, fname)])
    print('File list ready')

    outfile=open(dir+"/Outfile","w")

    for gt in groundtruth:
        f = open(gt[1])
        reader=csv.reader(f,delimiter="@")
        for row in reader:
            try:
                print(row[0])
                imagefilename=gt[0]+"/"+row[0]
                groundtruth=row[1]
                outfile.write(str(imagefilename)+"@"+str(groundtruth)+"\n")
            except:
                pass
        f.close()
    outfile.close()
    print("outputfile Ready")

def unicode_to_hex(dbfile,char):
    f=open(dbfile)
    line=f.readline()
    custom=["pn",1]
    while line:
        info=line.strip("\n").split(",")
        if(info[1]==char):
            if(len(info)==8):
                custom=[info[5],info[6],2]
            else:
                custom=[info[3],1]
            break
        line=f.readline()
    return custom

def load_bangla_to_custom(dbfile):
    map={}
    f=open(dbfile)
    line=f.readline()
    while line:
        info=line.strip("\n").split(",")
        map[info[0]]=info[-1]
        line=f.readline()
    print("Map loaded")
    return map

def makeh5_from_dir(imagedir,gtdir,outfile):
    hdf = h5py.File(outfile, "w")
    #bf=open("Bad_samples.txt","w")
    map=load_bangla_to_custom(dbfile)
    for root,sd,files in os.walk(gtdir):
        for fname in files:
            if(fname[-3:]=='txt'):
                f=open(os.path.join(root,fname))
                line=f.readline()
                while line:
                    info=line.strip("\n").split("@")
                    if(len(info)<2):
                        #bf.write(line+"\n")
                        line=f.readline()
                        continue
                    image_filename=imagedir+"/"+info[0]
                    groundtruth=info[1]
                    print("%s,groundtruth %s"%(image_filename,groundtruth))
                    #now read image
                    im = Image.open(image_filename).convert("L")
                    cols,rows=im.size
                    pixels=np.reshape(im.getdata(),[rows,cols])
                    #now assign groundtruth
                    unicode_line = convert_bangla_line_to_unicode(groundtruth)
                    unicode_compound_map = replace_compound_in_unicode_line(unicode_line, compounddbfile)
                    custom_line = convert_unicode_line_to_custom(unicode_compound_map, map)
                    reorderlist = ['m3', 'm8', 'm9']
                    reorder_line = reorder_modifier_in_custom_line(custom_line, reorderlist)
                    print("Reading ", fname, " target ", groundtruth, " Custsom ", reorder_line)
                    #now create hdf group
                    group = hdf.create_group(info[0])
                    group.create_dataset("Image", data=pixels)
                    group.attrs["Bangla_Target"] = groundtruth
                    group.attrs["Unicode_Target"]=unicode_line
                    group.attrs["Custom_Target"] = custom_line
                    group.attrs["Reorder_Target"] = reorder_line
                    group.attrs["SampleID"]=fname
                    #bf.write(image_filename+","+groundtruth+"\n")
                    line=f.readline()
    print("HDF Ready")
    hdf.close()
    #bf.close()


def find_character_histogram(charlist,h5file,statfile):
    #charlist contains all single and composite characters
    f=open(charlist)
    line=f.readline()
    map=[]
    bangla=[]
    charcount=0
    while line:
        info=line.strip("\n").split(",")
        print(info)
        map.append(info[-1])
        bangla.append(info[1])
        charcount+=1
        line=f.readline()
    f.close()
    print("All Character List Ready")
    hist=np.zeros([charcount])
    f=h5py.File(h5file)
    keys=list(f.keys())
    total=len(keys)
    for t in range(total):
        sample=f.get(keys[t])
        target=sample.attrs["Reorder_Target"]
        words=target.split("*")
        for w in words:
            chars=w.split()
            for ch in chars:
                try:
                    ind=map.index(ch)
                    hist[ind]+=1
                except:
                    pass
    f.close()
    f=open(statfile,"w")
    for i in range(charcount):
        f.write(bangla[i]+","+map[i]+","+str(hist[i])+"\n")
    f.close()
    print("Complete")

def append_single_characters(outfile,chardir,charfile):
    f=open(outfile,"a")
    for root,sd,files in os.walk(chardir):
        for fname in files:
            filename=os.path.join(root,fname)
            label=filename.split("/")[-2]
            fc=open(charfile)
            line=fc.readline()
            while line:
                line=line.strip("\n").split(",")
                if(label==line[0]):
                    unicode_tag=line[-1]
                    break
                line=fc.readline()
            fc.close()
            print(filename,"--",label,"--",unicode_tag)
            f.write(filename+"@"+unicode_tag+"\n")
    f.close()
    print("Completed")

def shuffle_outfile(outfile):
    filelist=[]
    f=open(outfile)
    line=f.readline()
    while line:
        filelist.append(line)
        line=f.readline()
    f.close()

    shuffle(filelist)

    print("List Shuffled")

    f=open(outfile,"w")
    for l in filelist:
        f.write(l+"\n")
    f.close()
    print("Outfile Updated")

def find_distinct_words(dir):
    all_words=[]
    for root,sd,files in os.walk(dir):
        for filename in files:
            abs_fname=os.path.join(root,filename)
            ftype=abs_fname[-3:]
            if(ftype=="txt"):
                #now read the file
                f=open(abs_fname)
                line=f.readline()
                while line:
                    line=line.strip("\n")
                    words=line.split()
                    all_words.extend(words)
                    line=f.readline()
                f.close()
                print("File ",abs_fname," Reading complete")
    wordset=list(set(all_words))
    f=open("Wordset.txt","w")
    for w in wordset:
        f.write(str(w)+"\n")
    f.close()
    print("Wordset Ready")

def find_distinct_words_hdf(h5file):
    f=h5py.File(h5file)
    keys=list(f.keys())
    total=len(keys)
    all_words=[]
    for t in range(total):
        sample=f.get(keys[t])
        target=sample.attrs['Bangla_Target']
        words=target.split()
        for w in words:
            all_words.append(w.replace(" ",""))
        print('Reading ',sample.attrs["SampleID"])
    f.close()
    return all_words

def find_distinct_words_allhdf(hdfs):
    all_words=[]
    for fn in hdfs:
        all_w=find_distinct_words_hdf(fn)
        all_words.extend(all_w)
        print('Reading ',fn)

    wordset = list(set(all_words))
    f = open("wordseth5.txt", "w")
    for w in wordset:
        f.write(w + "\n")
    f.close()

def test_sanity(hdffile,start,stop):
    f=h5py.File(hdffile)
    keys=list(f.keys())
    keys=keys[start:stop]
    for k in keys:
        sample=f.get(k)
        print(sample.name,sample.attrs['Bangla_Target'])

def crop_image(imagemat,threshold=250,savepath=None):
    #find 1st and last non white rows and 1st and last non white columns
    #imagemat is rows x colums => height x width numpy array
    original_w,original_h=imagemat.shape[1],imagemat.shape[0]
    print("original width=%d, height=%d"%(original_w,original_h))
    #finding top and bottom white rows
    top_flag=True
    bottom_flag=True
    last_top_white=0
    last_bottom_white=original_h-1
    for r in range(original_h):
        for c in range(original_w):
            pixval_top=imagemat[r][c]
            pixval_bottom=imagemat[original_h-r-1][c]
            if(top_flag)and(pixval_top<threshold):
                top_flag=False
                last_top_white=r
                #break
            if(bottom_flag)and(pixval_bottom<threshold):
                bottom_flag=False
                last_bottom_white=r
            if (not top_flag) and (not bottom_flag):
                break
        if(not top_flag)and(not bottom_flag):
            break
    print("Top %d Bottom %d"%(last_top_white,last_bottom_white))
    #finding left and right white rows
    left_flag=True
    right_flag=True
    last_left_white=0
    last_right_white=original_w-1
    for c in range(original_w):
        for r in range(original_h):
            pixval_left=imagemat[r][c]
            pixval_right=imagemat[r][original_w-c-1]
            if(left_flag)and(pixval_left<threshold):
                left_flag=False
                last_left_white=c
            if(right_flag)and(pixval_right<threshold):
                right_flag=False
                last_right_white=c
            if(not left_flag)and(not right_flag):
                break
        if (not left_flag) and (not right_flag):
            break
    print("Left %d Right %d" % (last_left_white, last_right_white))
    new_imagemat=imagemat[last_top_white:original_h-last_bottom_white,last_left_white:original_w-last_right_white]
    newimg=Image.fromarray(new_imagemat.astype('uint8')).convert('L')
    w,h=newimg.size[0],newimg.size[1]
    if(savepath is None):
        newimg.save('cropped.jpeg')
    else:
        newimg.save(savepath)
    return w,h


def test_crop_image(imfile):
    img=Image.open(imfile).convert('L')
    image_dim=img.size
    print(image_dim)
    imgmat=np.reshape(img.getdata(),[image_dim[1],image_dim[0]])
    crop_image(imgmat)

def crop_image_in_dir(imdir):
    #crops all image in imagedir and replace original image
    f=open("Croplog.txt",'w')
    for root,sd,filenames in os.walk(imdir):
        for fname in filenames:
            if(fname[-4:]=='jpeg'):#this is an image file
                abs_filename=os.path.join(root,fname)
                img = Image.open(abs_filename).convert('L')
                ow,oh = img.size[0],img.size[1]
                imgmat = np.reshape(img.getdata(), [oh, ow])
                nw,nh=crop_image(imgmat,savepath=abs_filename)
                msg="Cropped image %s,%d-%d,%d-%d"%(fname,ow,oh,nw,nh)
                print("Cropped image %s,%d-%d,%d-%d"%(fname,ow,oh,nw,nh))
                f.write(msg+"\n")
    f.close()



makeh5_from_dir("../TestGMMGC950_icdar2019","../GT","TestGMMGC950_icdar2019.h5") #Image files, corresponding gt, target filename
