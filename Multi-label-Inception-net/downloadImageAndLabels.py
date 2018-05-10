import wget
import json

path_images = './images/multi-label'
path_labels = './image_labels_dir'
filename = './train.json'

with open(filename) as json_data:
	data = json.load(json_data)

m = len(data['images'])

all_labels = []

MIN_LABEL = 1
MAX_LABEL = 100
count_downloaded = 0
count_skipped = 0

for image in data['images']:

	imageId = int(image['imageId'])
	
	if imageId > MAX_LABEL: break

	if imageId >= MIN_LABEL:

		url = image['url']
		wget.download(url, path_images + '/' + str(imageId) + '.jpg')
		count_downloaded += 1

print('\nDownloaded files:', count_downloaded)
print('\nSkipped files:', count_skipped)


	
