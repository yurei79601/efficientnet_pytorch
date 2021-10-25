"""
使用 EfficientNet 預測
"""
from PIL import Image
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from data_manager import image_preprocess_map
import config as cfg


def load_model(model_weight_path: str, device):
    """
    讀取訓練好的模型權重檔，並且實體化

    Args:
        model_weight_path: 模型權重檔路徑
        device: 使用運算的機器，cpu or gpu

    Returns:
        已讀取權重且分配好 device 的模型物件
    """
    model = EfficientNet.from_name(cfg.MODEL_NAME)
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, cfg.NUMBER_CLASSES)
    model_weight = torch.load(model_weight_path)
    model.load_state_dict(model_weight)
    model = model.to(device)
    return model


def image_preprocess(image: Image.Image) -> torch.Tensor:
    """
    預測前的影像預處理
    """
    image_map = image_preprocess_map()
    image_torch_batch = torch.unsqueeze(image_map(image), 0)
    return image_torch_batch


def inference_one(model, image: Image.Image, device):
    """
    使用 EfficientNet 預測單張圖片

    Args:
        model: 模型物件
        image: 影像檔，格式為 PIL.Image.Image
        device: 使用運算的機器，cpu or gpu

    Returns:
        predict_label: 預測 label (根據 cfg.PREDICT_LIST 轉換)
        predict_prob: 預測為 predict_label 的機率值
    """
    model.eval()
    image_torch = image_preprocess(image).to(device)
    print(image_torch.shape)
    with torch.no_grad():
        logits = model(image_torch)
    predict_index = torch.argmax(logits).item()
    predict_label = cfg.PREDICT_LIST[predict_index]
    predict_prob = torch.softmax(logits, dim=1)[0][predict_index].item()
    return predict_label, predict_prob
