# import systemwise package
import os
import threading
import queue

# import basic tools
import random

# import third-party package
import cv2

# import local tools in this project
import mudra_detect as md

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


# 設定結印圖片的路徑
image_path = 'images'
Mudra_dir_path = os.path.join(image_path, 'mudras')


# 設定各個狀態的旗標
end_game = False



target_queue_A = queue.Queue(maxsize=1)
target_queue_B = queue.Queue(maxsize=1)
answer_queue_A = queue.Queue(maxsize=5)
answer_queue_B = queue.Queue(maxsize=5)
end_game_queue = queue.Queue(maxsize=1)


playerA_t = threading.Thread(target=md.play, args=(0,'A', target_queue_A, answer_queue_A))
playerB_t = threading.Thread(target=md.play, args=(0,'B', target_queue_B, answer_queue_B))


while end_game == False:
    
    # 隨機產生指定忍術
    Ninjutsu_str   = random.choice(str().join(Ninjutsu_mudras_dict.keys()))


    # 根據忍術顯示第(1~5)張
        # 傳輸WebCam影像給神經網路
        # 神經網路回傳判斷機率
        # 機率在合格值以上時PASS，迴圈繼續下一張結印圖片
    for mudra_str in Ninjutsu_mudras_dict[Ninjutsu_str]:

        Mudra_img_path = os.path.join(Mudra_dir_path, (mudra_str + '.jpg'))
        if os.path.isfile(Mudra_img_path):
            Mudra_img = cv2.imread(Mudra_img_path)
            cv2.imshow(mudra_str, Mudra_img)
            cv2.waitKey(0)
        
        target_queue_A.put(mudra_str)
        target_queue_B.put(mudra_str)
    
    cv2.destroyAllWindows()
    

    # 顯示勝負結果

