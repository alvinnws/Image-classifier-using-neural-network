import csv
from multiprocessing import Manager, Process, freeze_support
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import sys

# Global variables -- Self explanatory names
DATAFOLER = "cropped-by-semantic-tag/_images_data.csv"
IMGPATH = "cropped-by-semantic-tag/"
IMG_HEIGHT = 175
IMG_WIDTH = 350
IMG_COUNT = 13319
NUM_CATEGORIES = 11
NUM_THREADS = 6
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
    # Open the csv with labels as reader
    f = open(DATAFOLER)
    reader = csv.reader(f)

    # Get images loaded into dictionary of Key: Value being Img: ID
    imgQueue = multiImportImage(IMG_COUNT)
    imgDict = {}
    for i in range(IMG_COUNT):
        try:
            (id, image) = imgQueue.get(timeout=1)
            imgDict[id] = image
        except:
            break

    # Create lists for images and the associated label
    dataImages = []
    dataLabels = []
    count = 0
    for row in reader:
        # Ignore first row
        if row[0] == "id":
            continue
        try:
            # Resize image to input size and add it and the label to each list
            resized = cv2.resize(imgDict[count], (IMG_HEIGHT, IMG_WIDTH))
            dataImages.append(resized)
            dataLabels.append(tf.keras.utils.to_categorical(CATDIC[row[1]], NUM_CATEGORIES))
            count += 1
        # Error occurs for images filtered out by multiImportImage(), skips the affected image
        except:
            count += 1
            continue
        
    # Split data into test and train categories
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(dataImages), np.array(dataLabels), test_size=0.4
    )

    # Retrieve model, print its shape using summary
    model = get_model()
    model.summary()

    # Train and evaluate
    model.fit(x_train, y_train, epochs=25)
    model.evaluate(x_test,  y_test, verbose=2)
        
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
        if id % 250 == 0:
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
def multiImportImage(IMG_COUNT):

    # Initialise Queues
    idQueue = Manager().Queue()
    imgQueue = Manager().Queue()
    for id in range(IMG_COUNT):
        idQueue.put(id)

    # Create and start threads, for loop just to allow minimising the lines in VSCode
    threads = []
    for i in range(NUM_THREADS):
        threads.append(Process(target=importImage, args=(idQueue, imgQueue)))
        threads[i].start()
    
    # Tell children no
    for i in range(NUM_THREADS):
        idQueue.put('STOP')

    # Wait for image loading to finish
    while not idQueue.empty():
        continue

    # Terminate multiprocessing, for loop yet again to enable minimising
    for i in threads:
        i.terminate()
    
    return imgQueue

# Make Model
def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(IMG_WIDTH,IMG_HEIGHT,3)),
        tf.keras.layers.Conv2D(32, 4, activation="relu"),
        tf.keras.layers.Conv2D(15, 4, activation="relu"),
        tf.keras.layers.Conv2D(30, 4, activation="relu"),
        tf.keras.layers.MaxPooling2D(4),       
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(300, activation="relu"),
        tf.keras.layers.Dense(300, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(500, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(200, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(120, activation="relu"),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

if __name__ == "__main__":
    # Multiprocessing documentation recommended to put freeze_support()
    freeze_support()
    main()
