#region     [分配並創建存放資料的dataset工作區資料夾]

# 原始的圖片位於./Source Pictures/train
# 資料集圖片位於./datasets/

import os, shutil

cwd = os.getcwd()
source_dir = os.path.join(cwd, 'source_data')
workspace_dir = os.path.join(cwd, 'workspace')

# 原始圖片存放的資料夾
original_dataset_dir = os.path.join(cwd, source_dir, 'images')
# 用來儲存少量資料集的目錄位置
dataset_dir = os.path.join(cwd, workspace_dir, 'datasets')
if not os.path.isdir(dataset_dir): os.mkdir(dataset_dir)  # 如果目錄不存在, 才建立目錄



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



#region     [分拆成訓練、驗證與測試目錄位置]

for purpose in purpose_dict:
    purpose_dir = os.path.join(dataset_dir, purpose)
    purpose_dict[purpose] = purpose_dir
    if not os.path.isdir(purpose_dir):    
        print("建立{}資料夾".format(purpose))
        os.mkdir(purpose_dir)

#endregion  [分拆成訓練、驗證與測試目錄位置]


#region     [依照label與purpose分配資料夾]

for purpose in purpose_dict:
    for label in label_list:
        purpose_label_dir = os.path.join(purpose_dict[purpose], label)
        if not os.path.isdir(purpose_label_dir):
            os.mkdir(purpose_label_dir)


#endregion  [依照label與purpose分配資料夾]



#endregion  [分配並創建存放資料的dataset工作區資料夾]


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
