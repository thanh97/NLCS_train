#export file csv ra thành ảnh
# import csv
# from PIL import Image
# import numpy as np
# import string
# import os
#
# csv_File_Path = '../data/A_Z_Data.csv'
# count = 1
# last_digit_Name = None
# image_Folder_Path = '../data/img_cut'
# Alphabet_Mapping_List = list(string.ascii_uppercase)
#
# for alphabet in Alphabet_Mapping_List:
#     path = image_Folder_Path + '\\' + alphabet
#     if not os.path.exists(path):
#         os.makedirs(path)
#
# with open(csv_File_Path, newline='') as csvfile:
#     reader = csv.reader(csvfile, delimiter=',', quotechar='|')
#     count = 0
#     for row in reader:
#         digit_Name = row.pop(0)
#         image_array = np.asarray(row)
#         image_array = image_array.reshape(28, 28)
#         new_image = Image.fromarray(image_array.astype('uint8'))
#
#         if last_digit_Name != str(Alphabet_Mapping_List[(int)(digit_Name)]):
#             last_digit_Name = str(Alphabet_Mapping_List[(int)(digit_Name)])
#             count = 0
#             print("")
#             print("Prcessing Alphabet - " + str(last_digit_Name))
#
#         image_Path = image_Folder_Path + '\\' + last_digit_Name + '\\' + str(last_digit_Name) + '-' + str(
#             count) + '.png'
#         new_image.save(image_Path)
#         count = count + 1
#
#         if count % 1000 == 0:
#             print("Images processed: " + str(count))

# # chuyển bộ data_full về ảnh array
# import shutil
# import os
# import cv2
# path   = "T:/data_goc/data_full/Z" #đường dẫn chứa thư mục cần chuển
# output = "T:/data/Z" #nơi lưu ảnh đã xử lý
# couter = 0
# for file_name in os.listdir(path):
#     path_file = "{}/{}".format(path, file_name)
#     if os.path.isfile(path_file):
#         # os.rename(path_file, path_file_res)
#         img = cv2.imread(path_file,0)
#         img_resize = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)  # resize img về kích thước 40x32
#         thresh = 172 # ngưỡng đặc trưng của ảnh
#         img_agray = cv2.threshold(img_resize, thresh, maxval=255, type=cv2.THRESH_BINARY_INV)[1] # chuyển ảnh về dạng trắng đen
#         # save img
#         cv2.imwrite("{}/{}".format(output,file_name),img_agray) # lưu ảnh

# đổi kích thước bộ dữ liệu csv lên 128x128
import shutil
import os
import cv2
path   = "T:/in" #đường dẫn chứa thư mục cần chuển
output = "T:/out" #nơi lưu ảnh đã xử lý
couter = 0
for file_name in os.listdir(path):
    path_file = "{}/{}".format(path, file_name)
    # path_file_res = "{}/new_{}".format(output, file_name)
    if os.path.isfile(path_file):
        # os.rename(path_file, path_file_res)
        img = cv2.imread(path_file)
        img_resize = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)  # resize img về kích thước 40x32
        # thresh = 172 # ngưỡng đặc trưng của ảnh
        # img_agray = cv2.threshold(img_resize, thresh, maxval=255, type=cv2.THRESH_BINARY_INV)[1] # chuyển ảnh về dạng trắng đen
        # save img
        cv2.imwrite("{}/{}".format(output,file_name),img_resize) # lưu ảnh

# #rename_file
# import os
# path = 'T:/data_goc/dataset/train/Z'
# i = 0
# for file_name in os.listdir(path):
#     path_file = "{}/{}".format(path, file_name)
#     i+=1
# print(i)
#     # if os.path.isfile(path_file):
#         # name = file_name.split(file_name[8])
#         # if  os.path.isfile(path_file):
#         #     os.rename(path_file, name[2])
#



# # loc du lieu
# import numpy as np
# import tensorflow as tf
# import tflearn
# from tflearn.layers.conv import conv_2d, max_pool_2d
# from tflearn.layers.core import input_data, dropout, fully_connected
# from tflearn.layers import regression
# import matplotlib.pyplot as plt
# import cv2
# import os
# # from tkinter import *
# # from tkinter import filedialog
# # mô hình
# IMG_SIZE = 28
# N_CLASSES = 26
#
# tf.reset_default_graph()
#
# network = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1])  # 1
#
# network = conv_2d(network, 32, 3, activation='relu')  # 2
# network = max_pool_2d(network, 2)  # 3
#
# network = conv_2d(network, 64, 3, activation='relu')
# network = max_pool_2d(network, 2)
#
# network = conv_2d(network, 32, 3, activation='relu')
# network = max_pool_2d(network, 2)
#
# network = conv_2d(network, 64, 3, activation='relu')
# network = max_pool_2d(network, 2)
#
# network = conv_2d(network, 32, 3, activation='relu')
# network = max_pool_2d(network, 2)
#
# network = conv_2d(network, 64, 3, activation='relu')
# network = max_pool_2d(network, 2)
#
# network = fully_connected(network, 1024, activation='relu')  # 4
# network = dropout(network, 0.8)  # 5
#
# network = fully_connected(network, N_CLASSES, activation='softmax')  # 6
# network = regression(network)
#
# model = tflearn.DNN(network)  # 7
#
# # load model va du doan
# model.load('./model_test/model2/mymodel.tflearn')
#
#
# path   = "T:/data_goc/I" #đường dẫn chứa thư mục cần chuển
# output = "T:/data_goc/I_dxl" #nơi lưu ảnh đã xử lý
# couter = 0
# for file_name in os.listdir(path):
#     path_file = "{}/{}".format(path, file_name)
#     img = cv2.imread(path_file,0)
#     img_resize= cv2.resize(img,(28,28),interpolation=cv2.INTER_AREA)
#     def prepare(img):
#             IMG_SIZE = 28
#             img_arr = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
#             new_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE), cv2.COLOR_BGR2GRAY)
#             return new_arr.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
#
#
#     alphabets_mapper = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K',
#                                 11: 'L',12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
#                                 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}
#
#     predict_img = model.predict(img_resize.reshape(-1, 28, 28, 1))
#     predict_img = np.argmax(predict_img, axis=-1)
#     kq = alphabets_mapper.get(predict_img[0])
#     if(kq =='I'):
#         cv2.imwrite("{}/{}".format(output,file_name),img) # lưu ảnh
#
#     else:
#         continue



