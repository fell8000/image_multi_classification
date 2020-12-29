# image_multi_classification

##COLAB을 이용하여 크기가 큰 데이터를 불러와, KERAS를 이용한 이미지 분류
##이미지 분류를 통한 예측


###라이브러리 import

'''PYTHON
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import experimental


from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint
'''


###COLAB을 이용한 이미지 전처리
