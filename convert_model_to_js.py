from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope
import keras
import tensorflowjs as tfjs

with CustomObjectScope({'relu6': keras.layers.ReLU(6.), 'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
    model = load_model("./models/10/weights.01.h5")
    # kien truc mo hinh
    # model.summary()
    names = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
             10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H", 18: "I", 19: "J",
             20: "K", 21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R", 28: "S", 29: "T", 30: "U",
             31: "V", 32: 'W', 33: "X", 34: "Y", 35: "Z"}

tfjs.converters.save_keras_model(model, "tjs_folder")
