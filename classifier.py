import csv
from multiprocessing import Manager, Process, freeze_support
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import sys

DATAFOLER = "cropped-by-semantic-tag/_images_data.csv"
IMGPATH = "cropped-by-semantic-tag/"
IMG_HEIGHT = 200
IMG_WIDTH = 400
NUM_CATEGORIES = 11
CATDIC = {
    "a": 0,
    "h1": 1,
    "h2": 2,
    "h3": 3,
    "h4": 4,
    "header": 5,
    "footer": 6,
    "form": 7,
    "input": 8,
    "button": 9,
    "textarea": 10
}

def main():
    f = open(DATAFOLER)
    reader = csv.reader(f)

    imageCount = 100
    f.seek(0)

    imgQueue = multiImportImage(imageCount)

    imgDict = {}
    for i in range(imageCount):
        try:
            (id, image) = imgQueue.get(timeout=1)
            imgDict[id] = image
        except:
            break
        
        '''cv2.imshow("", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows'''

    dataImages = []
    dataLabels = []
    count = 0
    for row in reader:
        if row[0] == "id":
            continue
        try:
            resized = cv2.resize(imgDict[count], (IMG_HEIGHT, IMG_WIDTH))
            dataImages.append(resized)
            dataLabels.append(tf.keras.utils.to_categorical(CATDIC[row[1]], 11))
            count += 1
        except:
            count += 1
            continue
        

    x_train, x_test, y_train, y_test = train_test_split(
        np.array(dataImages), np.array(dataLabels), test_size=0.4
    )

    model = get_model()

    model.summary()

    model.fit(x_train, y_train, epochs=10)
        
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
        if id % 25 == 0:
            print(str(id) + " images processed")
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
    idQueue = Manager().Queue()
    imgQueue = Manager().Queue()

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
def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(IMG_WIDTH,IMG_HEIGHT,3)),
        tf.keras.layers.Conv2D(32, 4, activation="relu"),
        tf.keras.layers.Conv2D(30, 4, activation="relu"),
        tf.keras.layers.Conv2D(30, 4, activation="relu"),
        tf.keras.layers.MaxPooling2D(5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(400, activation="relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(129, activation="relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(43, activation="relu"),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
# Train Model


if __name__ == "__main__":
    freeze_support()
    main()
