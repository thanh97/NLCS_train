import os
import shutil

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Khai bao bien mac dinh
MAX_IMAGES = 100

# Tao thu muc train
if not os.path.exists("T:/data_goc/ketqua/anh"):
    os.makedirs("T:/data_goc/ketqua/anh")

# # Tao thu muc validation
# if not os.path.exists("dataset/validation"):
#     os.makedirs("dataset/validation")

datagen = ImageDataGenerator(rotation_range=15,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             shear_range=0.01,
                             horizontal_flip=False,
                             fill_mode='nearest',
                             brightness_range=[0.5, 1.5])

# Doc du lieu
data_path = "T:/data_goc/dataset/train/I"
for dir_name in os.listdir(data_path):
    tmp_path = "{}/{}".format(data_path, dir_name)
    if not os.path.isdir(tmp_path):
        continue

    # Kiem tra va tao thu muc
    result_path = "T:/data_goc/ketqua/anh/{}".format(dir_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for file_name in os.listdir(tmp_path):
        file_path = "{}/{}".format(tmp_path, file_name)
        if not os.path.isfile(file_path):
            continue

        shutil.copy(file_path, result_path)
        # Lay du lieu
        img = load_img(file_path)
        img_array = img_to_array(img)
        img_array = img_array.reshape((1,) + img_array.shape)
        counter = 0
        for batch in datagen.flow(img_array,
                                  batch_size=1,
                                  save_to_dir=result_path,
                                  save_prefix=os.path.splitext(file_name)[0],
                                  save_format='png'):
            counter += 1
            if counter >= 5:
                break
