import processData

data_path='D:/Data/'

A=processData.prepData(data_path,510000,False)
A.readLabels('train')
A.generateHDF('train',47)