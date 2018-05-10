
import json
import numpy as np


def getMaxLabel(data):

	maxLabel = 0

	for labels in data['annotations']:
		for label in labels['labelId']:
			if int(label) > maxLabel:
				maxLabel = int(label)

	return maxLabel


def one_hot_multiple_outputs(v, num_classes):

	new_v = []

	for label in v:
		new_v.append(int(label))

	new_v = np.array(new_v)
	#new_v -= 1			# if element zero of array IS used

	return np.sum(np.eye(num_classes)[new_v], axis = 0).reshape(1, num_classes)


def decode_one_hot(v):

	result = []

	for i in range(len(v)):
		if v[i] == 1:
			result.append(i)

	return result


filename = './test_data/train_sample.json'

with open(filename) as json_data:
	data = json.load(json_data)

# Get number of training examples
m = len(data['annotations'])

# Get largest label in dataset
maxLabel = getMaxLabel(data)

# C = maxLabel   # if element zero of array IS used
C = maxLabel + 1 # if element zero of array IS NOT used	

Y_train = np.zeros((m, C))

for labels in data['annotations']:
	imageId = int(labels['imageId']) - 1
	Y_train[imageId] = one_hot_multiple_outputs(labels['labelId'], C)
	if imageId % 100000 == 0: print('%d / %d labels processed' % (imageId, m))

print('%d / %d labels processed' % (m, m))

''' FOR TESTING
for i in range(m):
	print(decode_one_hot(Y_train[i]))
'''

print('Training examples:', m)
print('Maximum label:', maxLabel)