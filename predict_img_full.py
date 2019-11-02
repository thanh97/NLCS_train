from keras.models import model_from_json
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.generic_utils import CustomObjectScope
import keras


with CustomObjectScope({'relu6': keras.layers.ReLU(6.), 'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
    model = load_model("./models/10/weights.01.h5")
    # kien truc mo hinh
    # model.summary()
    names = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
             10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H", 18: "I", 19: "J",
             20: "K", 21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R", 28: "S", 29: "T", 30: "U",
             31: "V", 32: 'W', 33: "X", 34: "Y", 35: "Z"}

# đọc ảnh trực tiếp và xử lý ảnh
img_red = cv2.imread('../data/img_test/8/1.jpg')
img = cv2.resize(img_red, (128, 128), interpolation=cv2.INTER_AREA)  # chuyển kích thước ảnh về 128x128
thresh = 172  # ngưỡng đặc trưng của ảnh, dươi ngưỡng pixel có màu đen và ngược lại là màu trắng.
img_agray = cv2.threshold(img, thresh, maxval=255, type=cv2.THRESH_BINARY_INV)[1]  # chuyển ảnh về dạng trắng đen
conver_img = img_to_array(img_agray) / 172
plt.imshow(conver_img)
plt.show()


# hàm xử lý ảnh dùng khi dự đoán nhiều ảnh
def prepare(filepath):
    IMG_SIZE = 128
    red_img = cv2.imread(filepath)
    img_resize = cv2.resize(red_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    thresh = 172
    img_agray = cv2.threshold(img_resize, thresh, maxval=255, type=cv2.THRESH_BINARY_INV)[1]
    conver_img = img_to_array(img_agray) / 172
    return conver_img.reshape(-1, IMG_SIZE, IMG_SIZE, 3)


# dự đoán
prediction = model.predict(conver_img.reshape(-1, 128, 128, 3))

# in ra mãng các giá trị dự đoán
print(prediction)

# lấy phần tử có giá trị lớn nhất
predict_img = np.argmax(prediction, axis=-1)

# in ra kết quả dự đoán
print(names.get(predict_img[0]))
