#region     [Import packages]

# import systemwise package
import os
import shutil
import queue
from queue import PriorityQueue as pq
import threading

# import basic tools
import random

# import third-party package
import cv2
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


#endregion  [Import packages]



#region     [Global variables]

# 設定
two_player = False




cwd = os.getcwd()
workspace_dir = os.path.join(cwd, 'workspace')


label_list = [
    "bird",
    "boar",
    "dog",
    "dragon",
    "horse",
    "monkey",
    "ox",
    "rabbit",
    "rat",
    "sheep",
    "snake",
    "tiger"
]

purpose_dict = {
    "train":"",
    "validation":"",
    "test":""
}


#region     [神經網路輸入的圖片尺寸]
img_x = 150
img_y = 150
#endregion  [神經網路輸入的圖片尺寸]


#region     [遊戲結束的Flag]
end_game = False
#endregion  [遊戲結束的Flag]

# 忍術結印表-字典
# generate by
# ````Python
# for i in range(20):
#     random.sample(label_list,5)
# ````
Ninjutsu_mudras_dict = {
    "A":['rabbit', 'tiger', 'dragon', 'horse', 'dog'],
    "B":['rabbit', 'tiger', 'horse', 'snake', 'sheep'],
    "C":['dog', 'sheep', 'rabbit', 'tiger', 'snake'],
    "D":['rat', 'ox', 'horse', 'dragon', 'dog'],
    "E":['monkey', 'ox', 'boar', 'tiger', 'rabbit'],
    "F":['snake', 'sheep', 'dog', 'bird', 'rabbit'],
    "G":['rat', 'dog', 'bird', 'tiger', 'monkey'],
    "H":['boar', 'horse', 'dog', 'bird', 'snake'],
    "I":['horse', 'tiger', 'rabbit', 'monkey', 'dog'],
    "J":['rat', 'dragon', 'dog', 'ox', 'boar'],
    "K":['boar', 'ox', 'rabbit', 'tiger', 'sheep'],
    "L":['boar', 'tiger', 'monkey', 'dog', 'sheep'],
    "M":['bird', 'ox', 'rabbit', 'horse', 'rat'],
    "N":['sheep', 'rabbit', 'monkey', 'dragon', 'bird'],
    "O":['sheep', 'dragon', 'monkey', 'ox', 'horse'],
    "P":['monkey', 'snake', 'dog', 'dragon', 'bird'],
    "Q":['ox', 'bird', 'monkey', 'rabbit', 'sheep'],
    "R":['bird', 'rabbit', 'dog', 'monkey', 'snake'],
    "S":['bird', 'dragon', 'ox', 'monkey', 'tiger'],
    "T":['ox', 'bird', 'monkey', 'sheep', 'boar']
}


# 忍術題目串列
Ninjutsu_List  = list()


# 設定結印圖片的路徑
image_path = 'images'
Mudra_dir_path = os.path.join(image_path, 'mudras')


# 神經網路模型存放路徑
model_dir = os.path.join(workspace_dir, 'train-logs')
best_model    = 'ninja-mudras-model'
best_model    = os.path.join(model_dir, best_model)

# 設定各個狀態的旗標
# global end_game = False


#endregion  [Global]



#region     [Declare Function]


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


def wait_queue_if_empty(q:queue):
    wait = q.empty()
    while wait:
        wait = q.empty()


def wait_queue_if_not_empty(q:queue):
    wait = q.empty()
    while not wait:
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

        print("detect")

        # 從WebCam讀取一張圖片
        success, frame = camera.read()
        # 成功read到圖片則顯示圖片
        if success:
            cv2.startWindowThread()
            cv2.imshow('frame', frame)
        
        cv2.waitKey(50)

        wait_queue_if_empty(target_queue)

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
            answer_queue.put(mudra_str)
            continue
        
        answer_queue.put('none')
    
    
    cv2.destroyWindow('frame')


def judge_mudra(target_queue:queue.Queue, answer_queue:queue.Queue):

    end_ninjutsu = False

    mudra_order = 0
    while end_ninjutsu == False:

        print("judge")

        wait_queue_if_empty(target_queue)
        target_mudra = target_queue.queue[0]
        wait_queue_if_not_empty(answer_queue)
        answer_mudra = answer_queue.get()

        if target_mudra == answer_mudra:
            mudra_order += 1
            target_queue.get()
        
        if mudra_order >= 5:
            end_ninjutsu = True


def play(webcam:int, player:str, target_queue:queue.Queue, answer_queue:queue.Queue):

    detect_mudra_thread = threading.Thread(target=detect_mudra, args=(webcam, player, target_queue, answer_queue))
    detect_mudra_thread.start()

    global end_game

    ninjutsu_complete_int = 0
    while end_game == False:

        judge_mudra(target_queue, answer_queue)
        ninjutsu_complete_int += 1

        if ninjutsu_complete_int >= 5:
            end_game = True

    detect_mudra_thread.join()


def show_target_mudra(target_queue:queue.Queue):
    # 根據忍術顯示第(1~5)張
        # 傳輸WebCam影像給神經網路
        # 神經網路回傳判斷機率
        # 機率在合格值以上時PASS，迴圈繼續下一張結印圖片

    global Ninjutsu_mudras_dict
    global Mudra_dir_path
    global Ninjutsu_List

    for Ninjutsu_str in Ninjutsu_List:

        for mudra_str in Ninjutsu_mudras_dict[Ninjutsu_str]:

            Mudra_img_path = os.path.join(Mudra_dir_path, (mudra_str + '.jpg'))
            print("show [", mudra_str, "] picture:")
            print(Mudra_img_path)
            # if os.path.isfile(Mudra_img_path):
            Mudra_img = cv2.imread(Mudra_img_path)
            # cv2.startWindowThread()
            cv2.imshow(Ninjutsu_str, Mudra_img)
            cv2.waitKey(500)

            target_queue.put(mudra_str)

            wait_queue_if_not_empty(target_queue)

        cv2.destroyWindow(Ninjutsu_str)


#endregion  [Declare Function]


model = tf.keras.models.load_model(best_model)

target_queue_A = queue.Queue(maxsize=1)
answer_queue_A = queue.Queue(maxsize=5)
if two_player:
    target_queue_B = queue.Queue(maxsize=1)
    answer_queue_B = queue.Queue(maxsize=5)


playerA_t = threading.Thread(target=play, args=(0,'A', target_queue_A, answer_queue_A))
playerA_t.start()
if two_player:
    playerB_t = threading.Thread(target=play, args=(1,'B', target_queue_B, answer_queue_B))
    playerB_t.start()

# while global end_game == False:

# 隨機產生5個指定忍術
for i in range(5):
    Ninjutsu_str = random.choice(str().join(Ninjutsu_mudras_dict.keys()))
    Ninjutsu_List.extend(Ninjutsu_str)

print("忍術題目：", Ninjutsu_List)

show_target_mudra_A_t = threading.Thread(target=show_target_mudra, args=(target_queue_A,))
show_target_mudra_A_t.start()
if two_player:
    show_target_mudra_B_t = threading.Thread(target=show_target_mudra, args=(target_queue_B,))
    show_target_mudra_B_t.start()


show_target_mudra_A_t.join()
playerA_t.join()
if two_player:
    show_target_mudra_B_t.join()
    playerB_t.join()

cv2.destroyAllWindows()


# 顯示勝負結果

