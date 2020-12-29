# image_multi_classification

## COLAB을 이용하여 크기가 큰 데이터를 불러와, KERAS를 이용한 이미지 분류와 예측


### 전제조건
파일 위치에 images 폴더가 있어야함.
tf.keras.preprocessing.image_dataset_from_directory를 사용하기 위해 카테고리 별로 파일이 나눠져 있어야함
ex)
images안 폴더 명 : 음식 실내 실외


### 라이브러리 import

```PYTHON
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import experimental


from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint
```


### COLAB을 이용한 이미지 전처리

```
    import os # miscellaneous operating system interfaces
    import shutil # high-level file operations

    batch_size = 32
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './images',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(300, 300),
    batch_size=batch_size,
    label_mode='categorical',
    class_names=["food", "interior", "exterior"]
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './images',
    validation_split=0.2,
    subset="validation",
    image_size=(300, 300),
    seed=123,
    batch_size=batch_size,
    label_mode='categorical',
    class_names = ["food", "interior", "exterior"]
    )

```
0.2만큼은 test를 위하여, 나머지 0.8만큼은 training을 위하여 이미지를 사용함.
