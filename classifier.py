import csv
import cv2
import numpy as np
import tensorflow as tensorflow
import sys

DATAFOLER = "cropped-by-semantic-tag/_images_data.csv"
IMGPATH = "cropped-by-semantic-tag/"

# Return True if image is blank (all pixels same colour), false otherwise
def blankSpace(img):
    sample = img[0][0]
    len1 = len(img)
    len2 = len(img[0])
    for i in range(len1):
        for j in range(len2):
            if i == j == 0:
                continue
            if (img[i][j] != sample).all():
                return False
    return True

# Load Data
def importImages(path):
    viable = []
    count = 0
    while True:
        print(count)
        try:
            # Load image into variable
            imagePath = path + str(count) + ".png"
            im = cv2.imread(imagePath, cv2.IMREAD_COLOR)

            # Find semantic tagged region
            Y, X = np.where(np.all(im==[0,0,255],axis=2))
            Y.sort()
            X.sort()

            if Y[0]+1 == Y[-1] or X[0]+1 == X[-1]:
                count += 1
                continue

            # Crop to semantic tagged region
            cropImg = im[Y[0]+1:Y[-1], X[0]+1:X[-1]]

            # If image contains no information, discard
            if blankSpace(cropImg):
                count += 1
                continue

            count += 1
            
        # After cv2 passes last image, return
        except:
            return viable

# Identify red box

# Crop to red box

# Identify images too large

# Make Model

# Train Model


if __name__ == "__main__":
    f = open(DATAFOLER)
    reader = csv.reader(f)

    images = importImages(IMGPATH)
    
    for i in images:
        cv2.imshow("", i)
        cv2.waitKey(0)
        cv2.destroyAllWindows

