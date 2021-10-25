"""
configuration setting
"""
import os


LR = 1e-3

# INPUT_SIZE = (height, width)
INPUT_SIZE = (32, 32)

ROTATE_ANGLE = 5
TRANSLATION = (0.05, 0.05)
MEAN_LIST = [0.485, 0.456, 0.406]
STD_LIST = [0.229, 0.224, 0.225]


MODEL_SAVE_PATH = os.path.expanduser("~/Desktop/My_file/study_讀書筆記/computer_vision/pytorch_notes/EfficientNet-PyTorch/model_weight")

MODEL_SAVE_NAME = "efficientnet_final.pth"

DATA_PATH = os.path.expanduser("~/Desktop/My_file/study_讀書筆記/computer_vision/pytorch_notes/CNN/MNIST_data")
# 需要分类的数目
NUMBER_CLASSES = 10
# 批次處理大小 (int)
BATCH_SIZE = 8
# 訓練多少個 epoch
EPOCH = 10
# True 為特徵提取，False 為微调
FEATURE_EXTACT = True
# 超参数设置
MODEL_NAME = "efficientnet-b0"

PREDICT_LIST = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
