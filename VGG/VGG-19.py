import tensorflow.keras as keras
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.applications.vgg19 import preprocess_input
from PIL import Image
from scipy import misc
from keras.optimizers import SGD
from os import listdir
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

def getModel(output_dim, input_shape):
    # output_dim: the number of classes (int)
    # input_shape: shape of the input data (tuple)

    vgg_model = VGG19(weights='imagenet', include_top=True, input_shape=input_shape)
    vgg_out = vgg_model.layers[-2].output  # Output of the second-to-last layer (before predictions)

    # Add a dropout layer
    vgg_out = Dropout(0.25)(vgg_out)

    # Create a new transfer learning model
    out = Dense(output_dim, activation="softmax", kernel_regularizer=l2(0.2))(vgg_out)

    tl_model = Model(inputs=vgg_model.input, outputs=out)

    # Freeze layers
    for layer in tl_model.layers[:-1]:
        layer.trainable = False

    # Compile the model
    tl_model.compile(loss="categorical_crossentropy", optimizer="adagrad", metrics=["acc"])
    print(tl_model.summary())

    return tl_model

# define functions to laod images
def loadBatchImages(path, s, nVal=2):
    # return array of images
    catList = listdir(path)
    loadedImagesTrain = []
    loadedLabelsTrain = []
    loadedImagesVal = []
    loadedLabelsVal = []

    for cat in catList[0:256]:
        deepPath = path + cat + "/"
        imageList = listdir(deepPath)
        indx = 0
        for images in imageList[0:s + nVal]:
            img = load_img(deepPath + images, target_size=(224, 224))  # Resize to (224, 224)
            img = img_to_array(img)
            img = preprocess_input(img)
            if indx < s:
                loadedLabelsTrain.append(int(images[0:3]) - 1)
                loadedImagesTrain.append(img)
            else:
                loadedLabelsVal.append(int(images[0:3]) - 1)
                loadedImagesVal.append(img)
            indx += 1

    return np.array(loadedImagesTrain), to_categorical(loadedLabelsTrain), \
        np.array(loadedImagesVal), to_categorical(loadedLabelsVal)


def shuffledSet(a, b):
    # shuffle the entire dataset
    assert np.shape(a)[0] == np.shape(b)[0]
    p = np.random.permutation(np.shape(a)[0])
    return (a[p], b[p])

model_name = "VGG_19_bs64_ep10"
print(model_name)
path = "/home/okancha/Keras-VGG16-TransferLearning/256_ObjectCategories/256_ObjectCategories/"
samCat = 32  # number of samples per category

data, labels, dataVal, labelsVal = loadBatchImages(path, samCat, nVal=2)


train = shuffledSet(np.asarray(data), labels)
val = shuffledSet(np.asarray(dataVal), labelsVal)


print(train[0].shape, val[0].shape)

output_dim = 257
input_shape = (224, 224, 3)
tl_model = getModel(output_dim, input_shape)


nb_epoch = 10
bs = 64

history = tl_model.fit(train[0], train[1], batch_size=bs, epochs=nb_epoch, validation_data=val, shuffle=True)

# Load the history from the JSON file
with open('/home/okancha/Keras-VGG16-TransferLearning/models/histories/' + model_name + ".json", 'w') as json_file:
    json.dump(history.history, json_file)

# Now, 'loaded_history' is a dictionary containing the saved training history

tl_model.save("/home/okancha/Keras-VGG16-TransferLearning/models/" + model_name + ".keras")


keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss for %d samples/ int' % samCat)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['val_acc'])
plt.title('model accuracy for %d samples per category' % samCat)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

tl_model.summary()
