import matplotlib.pyplot as plt
from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

# Tiền xử lý hình ảnh
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

# chuyển ma trận về thành 0 với 1 dễ cho việc tính toán trên ma trận
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Define
BATCH_SIZE = 300
N_CLASSES = 36
LR = 0.001
N_EPOCHS = 40
IMG_SIZE = 128

# thư mục ảnh training
train_generator = train_datagen.flow_from_directory(
    './dataset/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

# thư mục ảnh validation
validation_generator = test_datagen.flow_from_directory(
    './dataset/validation',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

# sử dụng model MobileNet để training
model = MobileNet(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=True, classes=N_CLASSES, weights=None)

# compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# kien truc mo hinh
model.summary()

# Check point
cp_callback = "./models/weights.{epoch:02d}.h5"
checkpoint = ModelCheckpoint(cp_callback, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Tiến hành huấn luyện mô hình
history = model.fit_generator(
    train_generator,
    steps_per_epoch=709595 // BATCH_SIZE,
    epochs=N_EPOCHS,
    validation_data=validation_generator,
    validation_steps=72000 // BATCH_SIZE,
    callbacks=callbacks_list)

# lưu mô hình
with open("./models/trongso.json", "w") as json_file:
    json_file.write(model.to_json())

# Biểu đồ training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
