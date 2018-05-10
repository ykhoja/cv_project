import json
import numpy as np

path_labels = './image_labels_dir'
filename = './train.json'

with open(filename) as json_data:
	data = json.load(json_data)

m = len(data['annotations'])

all_labels = []

MIN_LABEL = 1
MAX_LABEL = 15000

for labels in data['annotations']:
	imageId = int(labels['imageId'])
	
	if imageId > MAX_LABEL:
		break

	if imageId >= MIN_LABEL:
		with open(path_labels + '/' + str(imageId) + '.jpg.txt', 'w') as f:
			for label in labels['labelId']:
				f.write(label + '\n')
				if label not in all_labels: all_labels.append(label)

	if imageId % 100000 == 0: print('%d / %d labels processed' % (imageId, m))

with open('labels.txt', 'w') as f:
	for label in all_labels: f.write(label + '\n')

print('%d / %d labels processed' % (m, m))

print('Training examples:', m)
print('Number of labels:', len(all_labels))
print('Maximum label:', max(all_labels))
