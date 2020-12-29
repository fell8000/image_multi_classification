# image_multi_classification

## COLAB을 이용하여 크기가 큰 데이터를 불러와, KERAS를 이용한 이미지 분류와 예측


### 전제조건
>소스파일 위치에 images 폴더가 있어야함.

>colab을 사용하고 이미지 파일의 경우 구글드라이브에서 마운트 하므로 파일이 구글드라이브에 올라가있어야함.

>google drive에서 colab으로 마운트 하는과정은 인터넷을 참조 바람.

>tf.keras.preprocessing.image_dataset_from_directory를 사용하기 위해 카테고리 별로 파일이 나눠져 있어야함 
>>ex) images안 폴더 명 : food interior exterior


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

```PYTHON
    import os # miscellaneous operating system interfaces
    import shutil # high-level file operations

    batch_size = 32
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '/content/drive/"My Drive"/colab/images',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(300, 300),
    batch_size=batch_size,
    label_mode='categorical',
    class_names=["food", "interior", "exterior"]
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '/content/drive/"My Drive"/colab/images',
    validation_split=0.2,
    subset="validation",
    image_size=(300, 300),
    seed=123,
    batch_size=batch_size,
    label_mode='categorical',
    class_names = ["food", "interior", "exterior"]
    )

```
tf.keras.preprocessing.image_dataset_from_directory를 이용,
각각의 카테고리에서 정보들을 가져와 이미지 사이즈를 조정하고, label mode와 class_name으로 라벨의 개수를 multiple하게, 순서를 알파벳이 아닌 지정 순서로 변경한다.
이것은 후의 테스트를 위한 작업이며 생략 가능하다.
0.2만큼은 test를 위하여, 나머지 0.8만큼은 training을 위하여 이미지를 사용함.


### 모델 생성

```PYTHON
categories = ["food", "interior", "exterior"]
    with tf.device('/device:GPU:0'):
        model = Sequential([
            Dense(1024, activation='relu'),
            experimental.preprocessing.Rescaling(1. / 255),
            Input(shape=(300, 300, 3), name='input_layer'),
            Conv2D(64, (3, 3), padding="same", activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            Conv2D(32, (3, 3), padding="same", activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            Flatten(),
            Dense(64, activation='tanh'),
            Dense(3, activation='softmax', name='output_layer')
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_dir = '/content/drive/"My Drive"/colab/model'

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        model_path = model_dir + '/multi_img_classification.model'
        checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)

    history = model.fit(train_ds, validation_data=val_ds, batch_size=32, epochs=20,
                        callbacks=[checkpoint])
    model.save('model') #폴더
    plot_curve(history.history)
    print(history.history)
    print("train loss=", history.history['loss'][-1])
    print("validation loss=", history.history['val_loss'][-1])
```
COLAB에 GPU를 사용하여 데이터처리를 빠르게 하기위하여 사용.
CPU보다 CPU가 12배 정도의 속도가 이루어진다는 이야기 있음.
RESCALING의 경우 이미지 파일이 0~1사이의 값을 가지는것이 LOSS가 줄어드는 방법이라 이용.
3번의 CONV, MAXPOOLING의 과정을 거치며 저장됨.
history를 moddelfit할때 저장하며 epochs를 20으로 줘 정확도를 높인다. 
colab의 경우 한 번 epoch을 돌린 후 부터는 진행속도가 매우 빨라지는 장점이 있음.


