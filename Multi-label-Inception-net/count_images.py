import os

path = './image_originals/'

MIN_LABEL = 1
MAX_LABEL = 10000
count = 0


for i in range(MIN_LABEL, MAX_LABEL+1):
	if os.path.isfile(path + str(i) + '.jpg') == False:
		print (str(i)+'.jpg does not exist')
		count += 1

print('Missing files:', count)
print('Total files:', MAX_LABEL - MIN_LABEL + 1)