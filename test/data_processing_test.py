import numpy as np
import cv2
import glob

# Load all .jpg files in the test_data folder
filenames = glob.glob("./test_data/*.jpg")
filenames.sort()

# Load all images as numpy arrays
images = [(name, cv2.imread(name)) for name in filenames]

def display(img):
	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# Initialize X_train
size = 200	
num_examples = len(images)
X_train = np.zeros((num_examples, size, size, 3))

# Print the name and shape of each image
for m, (name, img) in enumerate(images):
	# print("Original image sizes")
	# print(name, ': ', img.shape)
	# display(img)

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

	# print("Padded image sizes")
	# print(name, ': ', img.shape)

	# Resize all images to have shape 100x100x3

	img = cv2.resize(img, (size, size))
	# print("Resized image sizes")
	# print(name, ': ', img.shape)
	# display(img)

	X_train[m] = img

print('X_train shape: ', X_train.shape)