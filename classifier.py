import csv
from multiprocessing import Manager, Process, freeze_support
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

# Optional Setting
IGNORETEXTAREA = False # Option to ignore images with label "textarea" as there are only 12, small sample size may hinder training
SAVEMODEL = False # If there is a desire to export the trained model
PLOTGRAPH = True # If there is a desire to plot a graph of the training

# Global variables -- Self explanatory names
LABELINFO = "cropped-by-semantic-tag/_images_data.csv" # Label file
IMGPATH = "cropped-by-semantic-tag/" # Folder with images
CPFILENAME = "save" # Folder name of model which will be saved
NUM_THREADS = 11 # 11 brings CPU util to 100% on my machine, change if necessary
IMG_COUNT = 13319 # Number of images to train and test with
EPOCHS = 200 # Number of epochs to train over
GRAPHAVERAGE = 3 # Odd numbers only, number of values to average over before plotting graph
# Recommended not to modify values beneath this comment
IMG_HEIGHT = 175
IMG_WIDTH = 350
NUM_CATEGORIES = 10 if IGNORETEXTAREA else 11
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
    f = open(LABELINFO)
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

    # Split data into test and train categories
    dataImages, dataLabels = listFromDict(imgDict, reader)        
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(dataImages), np.array(dataLabels), test_size=0.35
    )

    # Retrieve model, print its shape using summary
    model = get_model()
    model.summary()

    # Ability to toggle between saving the model or not, based on lowest validation loss
    if SAVEMODEL:
        checkpoint = "./checkpoints/" + CPFILENAME
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint,
            monitor="val_loss",
            mode="min",
            )

    # Train and evaluate
    history = model.fit(x_train, y_train,
            epochs=EPOCHS,
            validation_data=(x_test, y_test),
            callbacks=[model_checkpoint] if SAVEMODEL else None
            )
    model.evaluate(x_test,  y_test, verbose=2)

    # Plots graph of training if requested
    if PLOTGRAPH: showGraph(history, (GRAPHAVERAGE-1)/2)
    return

# Create lists for images and the associated label
def listFromDict(imgDict, reader):
    dataImages = []
    dataLabels = []
    count = 0
    for row in reader:
        # Ignore first row
        if row[0] == "id":
            continue
        try:
            if IGNORETEXTAREA and row[1] == "textarea":
                count+= 1
                continue
            # Resize image to input size and add it and the label to each list
            resized = cv2.resize(imgDict[count], (IMG_HEIGHT, IMG_WIDTH))
            dataImages.append(resized)
            dataLabels.append(tf.keras.utils.to_categorical(CATDIC[row[1]], NUM_CATEGORIES))
            count += 1
        # Error occurs for images filtered out by multiImportImage(), skips the affected image
        except:
            count += 1
            continue
    return dataImages, dataLabels

# Takes in a model history and n, plots a graph over epochs, averaged y values by 2n+1
# Skips first epoch as it tends to skew the graph
def showGraph(history, n):
    trainLoss = history.history["loss"]
    validLoss = history.history["val_loss"]
    validAccy = history.history["val_accuracy"]

    averagedTrainLoss, averagedValidLoss, averagedValidAccy = [], [], []
    for i in range(1, len(trainLoss)):
        added, midTLoss, midVLoss, midVAccy = 0, 0, 0, 0
        # Average over 2n+1
        for j in range(-n, n+1):
            try:
                midTLoss += trainLoss[i+j]
                midVLoss += validLoss[i+j]
                midVAccy += validAccy[i+j]
                added += 1
            except:
                continue
        averagedTrainLoss.append((midTLoss)/added)
        averagedValidLoss.append((midVLoss)/added)
        averagedValidAccy.append((midVAccy)/added)

    plt.plot(range(2,EPOCHS+1),averagedTrainLoss, label="Training Loss")
    plt.plot(range(2,EPOCHS+1),averagedValidLoss, label="Validation Loss")
    plt.plot(range(2,EPOCHS+1),averagedValidAccy, label="Validation Accuracy")
    plt.legend(loc='best')
    plt.show()
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
        if id % 1000 == 0:
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
        tf.keras.layers.Conv2D(28, 4, activation="relu"),
        tf.keras.layers.Conv2D(20, 4, activation="relu"),
        tf.keras.layers.MaxPooling2D(4),       
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(70, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(60, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(40, activation="relu"),
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
