import globalvar
import os
import shutil
import queue
import threading

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras.preprocessing import image
import pandas as pd
import numpy as np
import cv2

#region     [Global]
label_list = globalvar.label_list
purpose_dict = globalvar.purpose_dict
original_dataset_dir = globalvar.original_dataset_dir
dataset_dir = globalvar.dataset_dir
workspace_dir = globalvar.workspace_dir


batch_size_v = 50
img_x = globalvar.img_x
img_y = globalvar.img_y
#endregion  [Global]


def create_model():

    inputs = keras.Input(shape=(img_x, img_y, 3))
    x = layers.Conv2D( 64, (5,5), activation='relu')(inputs)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(32, (5,5), activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(128, (5,5), activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D( 64, (5,5), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(len(label_list),       activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(label_list), activation='softmax')(x) # 輸出神經元個數採用我們的label數量

    model = keras.Model(inputs, outputs, name='model')
    # model.summary()  # 查看模型摘要

    # %% [markdown]
    # 
    # ### 建立model儲存資料夾
    # 

    # %%
    model_dir = os.path.join(workspace_dir, 'train-logs')

    # model_dir 不存在時，建立新資料夾
    if not (os.path.isdir(model_dir)): 
        os.mkdir(model_dir)

    # %% [markdown]
    # 
    # ### 建立回調函數(Callback function)
    # 

    # %%
    # 將訓練紀錄儲存為TensorBoard的紀錄檔
    log_dir = os.path.join(model_dir, 'model')
    model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)

    # 儲存最好的網路模型權重
    best_model_h5 = 'ninja-mudras-model-best.h5'
    best_model_h5 = os.path.join(model_dir, best_model_h5)
    model_mckp = keras.callbacks.ModelCheckpoint(best_model_h5,
                                                monitor='val_categorical_accuracy',
                                                save_best_only=True,
                                                mode='max')


    # 問：改成多元分類是否要更改損失函數？
    # 答：參考他人專案改成 'categorical_crossentropy'
    model.compile(optimizer=optimizers.Adam(),
                loss=losses.CategoricalCrossentropy(),
                metrics=[metrics.CategoricalAccuracy()])

    model.load_weights(best_model_h5)
    
    return model


def wait_queue(q:queue):
    wait = q.empty()
    while wait:
        wait = q.empty()


def detect_mudra(webcam:int, player:str, target_queue:queue.Queue, answer_queue:queue.Queue):
    # Continuous detect mudra from webcam

    # Only admit mudra if the probability greater than 70%
    qualification = 0.7

    camera = cv2.VideoCapture(webcam)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH,  160)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

    global end_game
    while end_game == False:

        wait_queue(target_queue)

        # 從WebCam讀取一張圖片
        success, frame = camera.read()
        # 成功read到圖片則顯示圖片
        if success:
            cv2.imshow('frame', frame)


        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (img_x, img_y), interpolation=cv2.INTER_NEAREST)
        frame = np.array(frame) / 255
        frame = image.img_to_array(frame)
        frame = np.expand_dims(frame, axis = 0)

        global model
        predict_output = model.predict(frame)               # 輸出測試結果
        max_probablility_index = predict_output.argmax()    # 取得最高機率的index
        mudra_str = label_list[max_probablility_index]      # 根據index得到對應的mudra label
        max_probablility = np.max(predict_output)           # 取得最高機率數值

        if max_probablility > qualification:                # 達到標準才視為結印
            answer_queue.put()
            continue
        
        answer_queue.put('none')


def judge_mudra(target_queue:queue.Queue, answer_queue:queue.Queue):

    end_ninjutsu = False

    mudra_order = 0
    while end_ninjutsu == False:

        target_mudra = target_queue[0]
        answer_mudra = answer_queue.get()

        if target_mudra == answer_mudra:
            mudra_order += 1
            target_queue.get()
        
        if mudra_order >= 5:
            end_ninjutsu = True


def play(webcam:int, player:str, target_queue:queue.Queue, answer_queue:queue.Queue):

    detect_mudra_thread = threading.Thread(target=detect_mudra, args=(webcam, player, target_queue, answer_queue))

    global end_game

    ninjutsu_complete_int = 0
    while end_game == False:

        judge_mudra(target_queue, answer_queue)
        ninjutsu_complete_int += 1

        if ninjutsu_complete_int >= 5:
            end_game = True

model = create_model()
