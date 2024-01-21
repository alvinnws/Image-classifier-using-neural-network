import csv
from multiprocessing import Process, Queue, freeze_support
import cv2
import numpy as np
import tensorflow as tensorflow
import sys

DATAFOLER = "cropped-by-semantic-tag/_images_data.csv"
IMGPATH = "cropped-by-semantic-tag/"

def main():
    f = open(DATAFOLER)
    reader = csv.reader(f)

    imageCount = 200
    f.seek(0)

    imgQueue = multiImportImage(imageCount)

    return


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
def importImage(input, output):
    # Load image into variable
    for value in iter(input.get, 'STOP'):
        id = value
        print(id)
        imagePath = IMGPATH + str(id) + ".png"
        im = cv2.imread(imagePath, cv2.IMREAD_COLOR)

        # Find semantic tagged region
        Y, X = np.where(np.all(im==[0,0,255],axis=2))
        Y.sort()
        X.sort()

        # If image is not 0 pixels wide
        if not (Y[0]+1 == Y[-1] or X[0]+1 == X[-1]):

            # Crop to semantic tagged region
            cropImg = im[Y[0]+1:Y[-1], X[0]+1:X[-1]]

            # If image contains information, put in output
            if not blankSpace(cropImg):
                output.put((id, cropImg))

# Multiprocessing for intaking dataset
def multiImportImage(imageCount):
    idQueue = Queue()
    imgQueue = Queue()

    for id in range(imageCount):
        idQueue.put(id)

    # Create and start threads, for loop just to allow minimising the lines in VSCode
    for i in range(1):
        thread1 = Process(target=importImage, args=(idQueue, imgQueue))
        thread2 = Process(target=importImage, args=(idQueue, imgQueue))
        thread3 = Process(target=importImage, args=(idQueue, imgQueue))
        thread4 = Process(target=importImage, args=(idQueue, imgQueue))

        thread1.start()
        thread2.start()
        thread3.start()
        thread4.start()

    '''
    for i in range(imageCount):
        cv2.imshow("", imgQueue.get()[1])
        cv2.waitKey(0)
        cv2.destroyAllWindows'''
    
    # Tell children no
    for i in range(4):
        idQueue.put('STOP')

    # Wait for image loading to finish
    while not idQueue.empty():
        continue

    
    # Terminate multiprocessing, for loop yet again to enable minimising
    for i in range(1):
        thread1.terminate()
        thread2.terminate()
        thread3.terminate()
        thread4.terminate()
    
    return imgQueue

# Identify images too large

# Make Model

# Train Model


if __name__ == "__main__":
    freeze_support()
    main()
