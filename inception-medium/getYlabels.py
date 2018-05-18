# getYlabels.py extracts ground-truth information from train.json
# input: train.json
# outputs:
# 	- .jpg.txt files with labels for each image
#   - labels.txt file with a list of possible labels
#   - labels_count.txt file with count of each label through the dataset

import json
import numpy as np

path_labels = './image_labels_dir'
filename = './train.json'

with open(filename) as json_data:
	data = json.load(json_data)

m = len(data['annotations'])

all_labels = []
count_labels = {}

MIN_LABEL = 1
MAX_LABEL = 10000

for labels in data['annotations']:
	imageId = int(labels['imageId'])
	
	if imageId > MAX_LABEL:
		break

	if imageId >= MIN_LABEL:
		with open(path_labels + '/' + '0'*(7-len(str(imageId))) + str(imageId) + '.jpg.txt', 'w') as f:
			for label in labels['labelId']:
				f.write(label + '\n')
				if label not in all_labels: all_labels.append(label)

				# let's keep count of how frequent the labels are
				if label not in count_labels:
					count_labels[label] = 1
				else: count_labels[label] += 1


	if imageId % 100000 == 0: print('%d / %d labels processed' % (imageId, m))

# labels.txt contains the possible classes (one per line, not repeated)
with open('labels.txt', 'w') as f:
	for label in all_labels: f.write(label + '\n')

# labels_count.txt will have stats on class imbalance
with open('labels_count.txt', 'w') as f:
	for label in sorted(count_labels, key=count_labels.get, reverse=True):
		f.write('label: ' + label + ' count: ' + str(count_labels[label]) + '\n')  

print('%d / %d labels processed' % (m, m))

print('Training examples:', m)
print('Number of labels:', len(all_labels))
print('Maximum label:', max(all_labels))
