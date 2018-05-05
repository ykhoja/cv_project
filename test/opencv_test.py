import numpy as np
import cv2

# Folder with test images in it
path = './test_data/'

# Loading image '1.jpg', will display in a window 
# and wait for keystroke to close image
img = cv2.imread(path + '1.jpg')
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# The loaded image is a numpy array
print(img.shape)