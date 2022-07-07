# import the necessary packages
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

class Box:
    def __init__(self, x, y, w,h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

class Image:
    def __init__(self, name, faces_info):
        self.name = name
        self.bounding_box = []

        for i in range(len(faces_info)):
            self.bounding_box.append(Box(int(faces_info[i][0]), int(faces_info[i][1]), int(faces_info[i][2]), int(faces_info[i][3])))
        # print(self.bounding_box[0].x)
        # print(self.bounding_box[0].y)
        # print(self.bounding_box[0].w)
        # print(self.bounding_box[0].h)

images = []
path = "WiderSelected"
with open(os.path.join(path, "annotations.txt"), "r") as read_file:
    while True:
        image_name = read_file.readline().strip()
        if(image_name == ''):
            break 
        else:
            n = int(read_file.readline().strip())
            faces_info = []
            for i in range(n):
                faces_info.append(read_file.readline().strip().split(" "))
            images.append(Image(image_name, faces_info))
# split_index = int(3*len(images)/4)
# train_data = images[ : split_index]
# test_data = images[split_index : ]

##From example.py which mahdie sent
# initialize the list of data (images), our target output predictions
# (bounding box coordinates), along with the filenames of the
# individual images
data = []
targets = []
filenames = []
print("start process")
for image in images:
    imagePath = os.path.join(path, "train", image.name)
    org_image = cv2.imread(imagePath)
    # (h, w) = org_image.shape[:2]
    # scale the bounding box coordinates relative to the spatial
	# dimensions of the input image
    startX = float(image.bounding_box[0].x)
    startY = float(image.bounding_box[0].y)
    endX = float(image.bounding_box[0].x + image.bounding_box[0].w)
    endY = float(image.bounding_box[0].y + image.bounding_box[0].h)

	# load the image and preprocess it
    org_image = load_img(imagePath, target_size=(224, 224))
    org_image = img_to_array(org_image)

	# update our list of data, targets, and filenames
    data.append(org_image)
    targets.append((startX, startY, endX, endY))
    filenames.append(image.name)
print("end process")

# convert the data and targets to NumPy arrays, scaling the input
# pixel intensities from the range [0, 255] to [0, 1]
data = np.array(data, dtype="float32") / 255.0
targets = np.array(targets, dtype="float32")
# partition the data into training and testing splits using 90% of
# the data for training and the remaining 10% for testing
split = train_test_split(data, targets, filenames, test_size=0.10,
	random_state=42)
# unpack the data split
(trainImages, testImages) = split[:2]
(trainTargets, testTargets) = split[2:4]
(trainFilenames, testFilenames) = split[4:]
# write the testing filenames to disk so that we can use then
# when evaluating/testing our bounding box regressor
print("[INFO] saving testing filenames...")
test_path = "tests"
if(not os.path.isdir(test_path)):
    os.mkdir(test_path)
f = open(os.path.join(test_path, "test_filenames.txt") , "w")
f.write("\n".join(testFilenames))
f.close()