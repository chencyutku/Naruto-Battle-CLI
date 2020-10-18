#region     [分配並創建存放資料的dataset工作區資料夾]

# 原始的圖片位於./Source Pictures/train
# 資料集圖片位於./datasets/

import os, shutil

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
leave_game = False
#endregion  [遊戲結束的Flag]

#region     [對兩位玩家目前結印的判斷]
mudra_a = str()
mudra_b = str()
#endregion  [對兩位玩家目前結印的判斷]
