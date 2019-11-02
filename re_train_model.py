import keras
import matplotlib.pyplot as plt
from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from tflearn.layers.core import fully_connected
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras.models import load_model
from keras.layers import DepthwiseConv2D,ReLU
import tensorflowjs as tfjs
from keras.utils import CustomObjectScope



out_path="./tfjs_file"

# Training generator with augmentation
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode='nearest')

# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Define
BATCH_SIZE = 300
N_CLASSES = 36
LR = 0.001
N_EPOCHS = 6
IMG_SIZE = 128

train_generator = train_datagen.flow_from_directory(
    'T:/data_goc/dataset/train',  # this is the target directory
    target_size=(IMG_SIZE, IMG_SIZE),  # all images will be resized to 128x128
    batch_size=BATCH_SIZE,
    class_mode='categorical')

# This is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
    'T:/data_goc/dataset/validation',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

if __name__ == "__main__":
    # load model
    json_file = open("./models/7/trongso.json", "r")
    json_string = json_file.read()
    json_file.close()
    model = model_from_json(json_string)
    model = load_model("./models/7/weights.02.h5")

# kien truc mo hinh
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# # # create checkpoint
#
# cp_callback = "./models/weights.{epoch:02d}.h5"
# checkpoint = ModelCheckpoint(cp_callback, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint]
#
# # fit the model
# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=709595 // BATCH_SIZE,
#     epochs=N_EPOCHS,
#     validation_data=validation_generator,
#     validation_steps=72000 // BATCH_SIZE,
#     callbacks=callbacks_list)
#
# # Plot training & validation accuracy values
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
