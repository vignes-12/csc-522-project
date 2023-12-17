import tensorflow.keras as keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.optimizers import SGD
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
import json

from tensorflow.python.keras.callbacks import LearningRateScheduler


def get_model(output_dim, input_shape):
    # output_dim: the number of classes (int)
    # input_shape: shape of the input data (tuple)

    vgg_model = VGG16(weights='imagenet', include_top=True, input_shape=input_shape)
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


def shuffled_set(a, b):
    assert np.shape(a)[0] == np.shape(b)[0]
    p = np.random.permutation(np.shape(a)[0])
    return (a[p], b[p])

# Data augmentation
# datagen = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     preprocessing_function=preprocess_input
# )

model_name = "VGG_v2_lrsonly_bs32_ep10"
print(model_name)
path = "/home/okancha/Keras-VGG16-TransferLearning/256_ObjectCategories/256_ObjectCategories/"
samCat = 32  # number of samples per category

# Load data
# Load data with a 70%-30% split
data, labels, data_val, labels_val = loadBatchImages(path, samCat, nVal=2)
train = shuffled_set(np.asarray(data), labels)
val = shuffled_set(np.asarray(data_val), labels_val)

output_dim = 257
input_shape = (224, 224, 3)
tl_model = get_model(output_dim, input_shape)

# # Fine-tuning
# for layer in tl_model.layers[-6:]:
#     layer.trainable = True

# Compile the model again
tl_model.compile(loss="categorical_crossentropy", optimizer="adagrad", metrics=["acc"])

# Learning rate schedule
def step_decay(epoch):
    initial_lr = 0.01
    drop = 0.5
    epochs_drop = 5
    lr = initial_lr * (drop ** (epoch // epochs_drop))
    return lr

lr_scheduler = LearningRateScheduler(step_decay)

nb_epoch = 10
bs = 32

# Train the model with data augmentation and learning rate schedule
history = tl_model.fit(train[0], train[1], batch_size=bs, epochs=nb_epoch, validation_data=val, shuffle=True, callbacks=[lr_scheduler, keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')])
# history = tl_model.fit(
#     datagen.flow(train[0], train[1], batch_size=bs),
#     steps_per_epoch=len(train[0]) // bs,
#     epochs=nb_epoch,
#     validation_data=(val[0], val[1]),
#     shuffle=True,
#     callbacks=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
# )

with open('/home/okancha/Keras-VGG16-TransferLearning/models/histories/' + model_name + ".json", 'w') as json_file:
    # Convert numpy arrays with float32 to regular Python floats
    history_history_serializable = {key: [float(val) for val in value] if isinstance(value, np.ndarray) else value for
                                    key, value in history.history.items()}

    # Save the serialized history to a JSON file
    json.dump(history_history_serializable, json_file)

# Now, 'loaded_history' is a dictionary containing the saved training history

tl_model.save("/home/okancha/Keras-VGG16-TransferLearning/models/" + model_name + ".keras")


# keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss for %d samples/ int' % samCat)
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
#
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy for %d samples per category' % samCat)
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.show()
#
# tl_model.summary()

# ... (the rest of your code)
