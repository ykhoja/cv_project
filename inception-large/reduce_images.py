import numpy as np
import cv2
import glob

# Load all .jpg files in the test_data folder
filenames = glob.glob("./image_originals/*.jpg")
filenames.sort()

def display(img):
	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

size = 299  # Inception v3 was trained on ImageNet using images of size 299 x 299 x 3
num_examples = len(filenames)

for m, name in enumerate(filenames):
	img = cv2.imread(name) # Load one image at a time

    # Ensure that each image is square
	H, W, _ = img.shape
	if W > H:
		delta = (W - H) // 2
		residual = (W - H) % 2 != 0
		img = cv2.copyMakeBorder(img, delta + residual, delta, 0, 0, cv2.BORDER_CONSTANT, 0 )
	elif W < H:
		delta = (H - W) //2
		residual = (H - W) % 2 != 0
		img = cv2.copyMakeBorder(img, 0, 0, delta + residual, delta, cv2.BORDER_CONSTANT, 0)

	# Resize all images to have shape )size x size x 3)

	img = cv2.resize(img, (size, size))

	cv2.imwrite('./images/multi-label/' + '0'*(7-len(str(m+1))) + str(m+1) + '.jpg', img)
	print(str(m) + ' files reduced')

	