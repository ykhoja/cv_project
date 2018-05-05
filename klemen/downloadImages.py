import processData

data_path='D:/Data/'

A=processData.prepData(data_path)
A.downloadImages(A.readXControl('train'),598800)