import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import experimental


from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping


def plot_loss_curve(history):
    plt.figure(figsize=(15, 10))
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()


def predict_image_sample(model, val_ds,num=-1):
    if num<0:
        from random import randrange
        test_sample_id = randrange(9000)
        
        
    yy=[]
    result=[]
    target_name=["food","interior","exterior"]
    img=[]
    
    
    n=0
    for x,y in val_ds: #32장 복사
        if(n==num):
            n+=1
            continue
        yy=y #y_actual
        img=x #img
        result=model.predict_on_batch(x) #y_predict
        from sklearn.metrics import classification_report
        break
    
    
    import cv2
    img=np.array(img)
    img/=255.0
    
    yy=np.array(yy)
    result=np.array(result)
    
    
    
    
    for i in range(32):
        b, g, r = cv2.split(img[i])   # img파일을 b,g,r로 분리
        img[i] = cv2.merge([r,g,b]) # b, r을 바꿔서 Merge
        cv2.imshow("g",img[i])
        cv2.waitKey(0)
        #test_image = img[i].reshape(1,300,300,3)
        
        y_actual = yy[i]
        print("y_actual number=", y_actual)
        if(y_actual[0]==1):
            print("실제 : 음식")
        elif(y_actual[1]==1):
            print("실제 : 실내")
        elif(y_actual[2]==1):
            print("실제 : 실외")
        
        
        y_pred = result[i]
        print(y_pred)
        y_pred = np.argmax(y_pred, axis=0)
        if(y_pred==0):
            print("예측 : 음식")
        elif(y_pred==1):
            print("예측 : 실내")
        elif(y_pred==2):
            print("예측 : 실외")
        


img_height=300
img_width=300

def make_image_list():
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



    categories = ["food", "interior", "exterior"]
    with tf.device('/gpu:0'):
        model = Sequential([
            experimental.preprocessing.Rescaling(1. / 255),
            Input(shape=(300, 300, 3), name='input_layer'),

            Conv2D(64, kernel_size=3, padding="same", activation='relu',name='conv1'),
            MaxPooling2D(pool_size=2),

            #tf.keras.layers.Dropout(0.25),
            Conv2D(128, kernel_size=3, padding="same", activation='relu',name='conv2'),
            MaxPooling2D(pool_size=2),
            
            Conv2D(256, kernel_size=3, padding="same", activation='relu',name='conv3'),
            MaxPooling2D(pool_size=2),

            Flatten(),
            tf.keras.layers.Dropout(0.25),
            Dense(256, activation='relu'),
            Dense(3, activation='softmax', name='output_layer')
        ])
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_dir = '/content/drive/MyDrive/colab/'



        history = model.fit(train_ds, validation_data=val_ds, epochs=8)
        model.summary()
    model_path = model_dir + '/multi_img_classification.model'                
    model.save('model-201711299')
    plot_curve(history.history)
    print(history.history)
    print("train loss=", history.history['loss'][-1])
    print("validation loss=", history.history['val_loss'][-1])



def plot_curve(history):
    y_vloss = history['val_loss']
    y_loss = history['loss']
    x_len = np.arange(len(y_loss))

    plt.plot(x_len, y_vloss, marker='.', c='red', label='val_set_loss')
    plt.plot(x_len, y_loss, marker='.', c='blue', label='train_set_loss')

    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid()
    plt.show()
    

def evaluate_mine(model,val_ds):
    from sklearn.metrics import classification_report
    yy=[]
    result=[]
    
    y_test=[]
    y_=[]
    for x,y in val_ds: #32장 복사
        yy=y #y_actual
        result=model.predict_on_batch(x) #y_predict
        yy=np.array(yy)
        result=np.array(result)
        for i in range(32):
            y_pred = result[i]
            y_pred = np.argmax(result[i], axis=0)
            if(y_pred==0):
                result[i]=[1,0,0]
            elif(y_pred==1):
                result[i]=[0,1,0]
            elif(y_pred==2):
                result[i]=[0,0,1]
            if i==1:
                y_test=yy[i]
                y_=result[i]
            else:
                np.append(y_test,yy[i])
                np.append(y_,result[i])
    
    target_name=['food','interior','exterior']
    print(classification_report(y_test, y_, target_names=target_name))   
        
    
    

if __name__ == '__main__':
    '''
    #폴더 나누기 작업
    import os, glob, numpy as np

    caltech_dir = "./images"
    categories = ["food", "interior", "exterior"] 
    for idx, cat in enumerate(categories):
        files = glob.glob(caltech_dir + "/" + cat + "*.jpg")
        print(cat, " 파일 길이 : ", len(files))
        for i, f in enumerate(files):
            #파일 이동
            shutil.move("./images/interior/"+"food"+str(i+1)+".jpg", "./images/interior/" + "interior"+str(i+1)+".jpg")

    '''

    #모델 생성 함수
    #make_image_list()
    
    #모델 로드
    model=load_model('./model-201711299')
        


    batch_size = 32
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './images',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(300, 300),
    #batch_size=batch_size,
    label_mode='categorical',
    class_names=["food", "interior", "exterior"]
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    './images',
    validation_split=0.2,
    subset="validation",
    image_size=(300, 300),
    seed=123,
    #batch_size=batch_size,
    label_mode='categorical',
    class_names = ["food", "interior", "exterior"]
    )
    
    #predict vs actual 32장씩 (batch size)
    predict_image_sample(model, val_ds,2)


    #자체 성능평가
    #evaluate_mine(model,val_ds)



    
    
    