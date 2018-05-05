#template how to use the class to get mini_batches

import processData

data_path='C:/Users/ykhoja/cs231n/cv_project/data/'

#initialize data; sets the data path and opens the h5 tables
A=processData.prepData(data_path, max_record = 100)

#random initialize idx to pick the records
A.initMiniBatch(no_recs=64,train=70,dev=20,test=10) #train/dev/test are in % of total
EOF=0
while EOF==0:
    EOF,X,Y=A.getMiniBatch('train') #every call delivers one mini batch

print('X = ', X.shape) # X =  (30000, 7)
print('Y = ', Y.shape) # X =  (228, 7)