import processData

data_path='D:/Data/'

A=processData.prepData(data_path)
A.X_to_int('train',False,110001)