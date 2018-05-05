import numpy as np
import json
import csv
from PIL import Image
import requests
from io import BytesIO
import tables
import pandas as pd
from itertools import islice

class prepData(object):

    def __init__(self,data_path,max_record=500000,nongen_mode=True):
        self.data_path=data_path
        self.mini_batch_counter=0
        self.max_record=max_record
        self.mini_batch=dict()
        self.mini_batch['train']=list()
        self.mini_batch['dev']=list()
        self.mini_batch['test']=list()
        self.mini_batch_size=0
        if nongen_mode:
            self.db_X = tables.open_file(self.data_path+'X.h5', 'r')
            self.db_Y = tables.open_file(self.data_path+'Y.h5', 'r')
            self.t_rng1=self.db_X.get_node('/','train_0')
            self.t_rng2=self.db_X.get_node('/','train_6000')
            self.t_rng3=self.db_X.get_node('/','train_12000')
            self.t_rng4=self.db_X.get_node('/','train_18000')
            self.t_rng5=self.db_X.get_node('/','train_24000')
            self.t_Y=self.db_Y.get_node('/','labels')

    def readLabels(self,filename):
        with open(self.data_path+filename+'.json') as json_train:
            d=json.load(json_train)

        annotationsF=d['annotations']

    #    get dimensions for numpy array and initialize
        m=0
        noLabels=0
        Y=pd.DataFrame()
        for l in annotationsF:
            for k,v in l.items():
                if k=='imageId':
                    if m<int(v): m=int(v)
                if k=='labelId':
                    for label in v:
                        if noLabels<int(label): noLabels=int(label)
                        
        #Y
        npyY=np.zeros((m,noLabels),dtype=np.int)        
        for l in annotationsF:
            for k,v in l.items():
                if k=='imageId':
                    m=int(v)
                if k=='labelId':
                    for label in v:
                        npyY[m-1][int(label)-1]=1
            if int(l['imageId'])%100==0:
                print('label of img: '+ l['imageId'])

        Y=pd.DataFrame(data=npyY,index=range(1,m+1),columns=list(range(1,noLabels+1)))
        Y.to_hdf(self.data_path+'Y.h5',key='labels',mode='a',format='t')

    def readXControl(self,filename):
    #get features
        X_control=list()
        try:
            with open(self.data_path+'X_'+filename+'_control.csv','r') as csvfile:
                w = csv.reader(csvfile,delimiter=',')
                for row in w:
                    if len(row)>0: X_control.append(row)
                csvfile.close()
        except:
            with open(self.data_path+filename+'.json') as json_train:
                d=json.load(json_train)
                
            imagesF=d['images']
    
        #X
            X_control=list()
            for l in imagesF:
                X_control.append([l['imageId'],l['url']])
            
            with open(self.data_path+'X_'+filename+'_control.csv','w') as csvfile:
                w = csv.writer(csvfile,delimiter=',')
                for l in X_control:
                    w.writerow(l)
                csvfile.close()
            imagesF=None
            
        return X_control
    
    def downloadImages(self,X_control,start_no=0,end_no=1000000):
        #get image
        X=np.array([], dtype=int)

        for l in X_control:
            if int(l[0])>start_no and int(l[0])<=end_no:
                print('processing:',l[0])
                xi=np.array([],dtype=int)
                
                response = requests.get(l[1])
                img = Image.open(BytesIO(response.content))

                #resize    
                imgsize=img.size
                if imgsize[0]>imgsize[1]:
                    newsize=(100,int(100/imgsize[0]*imgsize[1]))
                else:
                    newsize=(int(100/imgsize[1]*imgsize[0]),100)
                img=img.resize(newsize)
            
                #vectorize    
                img_list=list(img.getdata())
                xi=np.asarray(img_list)
                xi=xi.reshape(img.size[0],img.size[1]*3)
                zxi=np.zeros((100,300), dtype=int)

                zxi[int((100-img.size[0])/2):int((100+img.size[0])/2),\
                    int((300-img.size[1]*3)/2):int((300+img.size[1]*3)/2)]=xi
                
                zxi=zxi.reshape(1,-1)
                zxi=np.insert(zxi,0,int(l[0]),axis=1)
                X=np.append(X,zxi)

                if int(l[0])%100==0:
                    fileno=int((int(l[0])-1)/10000)*10000
                    X=X.reshape(-1,30001)
                    with open(self.data_path+'X_train_'+str(fileno)+'.csv','a+') as csvfile:
                        w = csv.writer(csvfile,delimiter=',',lineterminator='\n')
                        try:
                            w.writerows(X.tolist())
                        except OSError:
                            try:
                                w.writerows(X.tolist())
                            except OSError:
                                w.writerows(X.tolist())
                        csvfile.close()
                        
                    X=np.array([], dtype=int)

    def X_to_int(self,filename,writeNPY=False,maxRecord=2000000):
        with open(self.data_path+'X_'+filename+'.txt','r') as txtfile:
            txtr = csv.reader(txtfile,delimiter=',')
            X=np.array([],dtype=int)
            i=1
            fileno=100000
#            for row in txtr:
            for row in islice(txtr, 100000, maxRecord):
                if int(float(row[0]))%100==0:
                    print('processing:', int(float(row[0])))
                xi=np.array([], dtype=int)
                xi=np.asarray([int(float(x)) for x in row])
                xi=xi.reshape(1,30001)
                if writeNPY:
                    xi=xi[0,1:].reshape(-1,30000)
                    if X.size==0:
                        X=xi.T
                        X=X.reshape(-1,1)
                    else:
                        X=np.append(X,xi.T, axis=1)
                    if i%10000==0:
                        np.save(self.data_path+'X_'+filename+'_'+str(i)+'.npy',X)
                        X=np.array([],dtype=int)
                else:
                    if (int(float(row[0]))-1)%10000==0:
                        fileno=int(float(row[0]))-1                    
#                    X=np.append(X,xi)
                    with open(self.data_path+'X_train_'+str(fileno)+'.csv','a+') as csvfile:
                        w = csv.writer(csvfile,delimiter=',',lineterminator='\n')
                        try:
                            w.writerow(xi.tolist())
                        except OSError:
                            try:
                                w.writerow(xi.tolist())
                            except OSError:
                                w.writerow(xi.tolist())
                        csvfile.close()
                i+=1
                
            txtfile.close()
    
    def initMiniBatch(self,no_recs=64,train=70,dev=20,test=10):
        self.mini_batch_size=no_recs
        v=np.random.randint(0,100,size=self.max_record)
        i=0
        for x in v:
            if x<train:
                self.mini_batch['train'].append(i)
            if x>=train and x<(train+dev):
                self.mini_batch['dev'].append(i)
            if x>=(train+dev):
                self.mini_batch['test'].append(i)
            i+=1

    def getMiniBatchIdx(self,lot='train'):        
        idx=self.mini_batch[lot][self.mini_batch_counter*self.mini_batch_size:(self.mini_batch_counter+1)*self.mini_batch_size]
        return(idx)

    def getImage(self,image_no):
        image = list()
        image.extend(self.t_rng1.table[image_no][1].tolist())
        image.extend(self.t_rng2.table[image_no][1].tolist())
        image.extend(self.t_rng3.table[image_no][1].tolist())
        image.extend(self.t_rng4.table[image_no][1].tolist())
        image.extend(self.t_rng5.table[image_no][1].tolist())
        return(np.asarray(image).reshape(1,-1),self.t_Y.table[image_no][1].reshape(1,-1))


    def getMiniBatch(self,lot='train'):
        idx=self.getMiniBatchIdx(lot)
        idx.sort()
        X=np.array([])
        Y=np.array([])
        for i in idx:
            xi,yi=self.getImage(i)
            if X.size==0:
                X=xi
                Y=yi
            else:
                X=np.append(X,xi,axis=0)
                Y=np.append(Y,yi,axis=0)
        X=X.T
        Y=Y.T

        self.mini_batch_counter+=1
        EOD=0
        if self.mini_batch_counter*self.mini_batch_size>self.max_record:
            EOD=1
        return(EOD,X,Y)


    def generateHDF(self, lot='train',start_no=0):
        for n in range(start_no,int(self.max_record/10000)):
            fileno=n*10000
            filename=self.data_path+'X_'+lot+'_'+str(fileno)+'.csv'
            Xdf=pd.read_csv(filename,delimiter=',',header=None, names=np.array(range(0,30000)),index_col=0,dtype=int)
            for i in range(0,5):
                print('processing:'+str(n)+' c-group:'+str(i))
                Xdf.iloc[:,i*6000:(i+1)*6000].to_hdf(self.data_path+'X.h5',key=lot+'_'+str(i*6000),mode='a',format='t',append=True)

