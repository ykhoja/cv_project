import numpy as np
import json
import csv
from PIL import Image
import requests
from io import BytesIO
import tables
import pandas as pd

#data_path='G:/Team Drives/CS230 CS231n Team Drive/Data/'
data_path='/Users/kncas/Google Drive File Stream/Team Drives/CS230 CS231n Team Drive/Data/'

class prepData(object):

    def __init__(self,data_path):
        self.data_path=data_path

    def readLabels(self,filename):
        try:
            Y=np.array([])
            Y=np.load(self.data_path+filename+'.npy')
        except IOError:
            with open(self.data_path+filename+'.json') as json_train:
                d=json.load(json_train)

            annotationsF=d['annotations']

        #    get dimensions for numpy array and initialize
            m=0
            noLabels=0
            for l in annotationsF:
                for k,v in l.items():
                    if k=='imageId':
                        if m<int(v): m=int(v)
                    if k=='labelId':
                        for label in v:
                            if noLabels<int(label): noLabels=int(label)
            Y=np.zeros((m,noLabels),dtype=np.int)

            #Y
            for l in annotationsF:
                for k,v in l.items():
                    if k=='imageId':
                        m=int(v)
                    if k=='labelId':
                        for label in v:
                            Y[m-1][int(label)-1]=1
            np.save(self.data_path+filename+'.npy',Y)
            annotationsF=None
            d=None
        return Y        

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
    
    def downloadImages(self,X_control,start_no=0):
        #get image
        X=np.array([], dtype=int)

        for l in X_control:
            if int(l[0])>start_no:
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
                    X=X.reshape(-1,30001)
                    with open(self.data_path+'X_train.csv','a+') as csvfile:
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

    def X_to_int(self,filename):
        with open(self.data_path+'X_'+filename+'.txt','r') as txtfile:
            txtr = csv.reader(txtfile,delimiter=',')
            X=np.array([],dtype=int)
            i=1
            fileno=1
            for row in txtr:
                xi=np.array([], dtype=int)
                xi=np.asarray([int(float(x)) for x in row])
#                X=np.append(X,xi)
                if i%100==0:
                    print('processing:', i)
                with open(self.data_path+'X_train_'+str(fileno)+'.csv','a+') as csvfile:
                    w = csv.writer(csvfile,delimiter=',',lineterminator='\n')
                    try:
                        w.writerow(xi)
                    except OSError:
                        try:
                            w.writerow(xi)
                        except OSError:
                            w.writerow(xi)
                    csvfile.close()
                if i%10000==0:
                    fileno=i
#                    i=0
#                    X=X.reshape(-1,30001)
#                    with open(self.data_path+'X_train_'+str(X[0][0])+'.csv','a+') as csvfile:
#                        w = csv.writer(csvfile,delimiter=',',lineterminator='\n')
#                        try:
#                            w.writerows(X.tolist())
#                        except OSError:
#                            try:
#                                w.writerows(X.tolist())
#                            except OSError:
#                                w.writerows(X.tolist())
#                        csvfile.close()
#                    X=np.array([])
                i+=1
                
            txtfile.close()
        


#run the stuff
A=prepData(data_path)
#A.readLabels('train')
#A.downloadImages(A.readXControl('train'),95272)
A.X_to_int('train')
