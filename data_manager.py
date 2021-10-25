"""
functions for data generation
"""
import os
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data.dataloader import default_collate
import config as cfg


def my_collate_fn(batch):
    """
    batch 中每個元素形如 (data, label)
    """
    # 過濾為 None 的資料
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0:
        return torch.Tensor()
     # 用默認方式拼接過濾後的 batch 資料
    return default_collate(batch)


def image_preprocess_map():
    image_map = transforms.Compose(
        [
            transforms.Resize(cfg.INPUT_SIZE),
            transforms.CenterCrop(cfg.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.MEAN_LIST, std=cfg.STD_LIST),

        ]
    )
    return image_map


def get_data_transforms(input_size: tuple) -> dict:
    """
    定義訓練時，讀取資料的預處理程序，其流程有
        1. resize
        2. 中心化
        3. rotate: 使用 cfg.ROTATE_ANGLE，單位是「度」
        4. 平移: 使用 cfg.TRANSLATION
        5. 變成 torch.Tensor 的格式
        6. normalize: 使用 cfg.MEAN_LIST 與 cfg.STD_LIST

    Args:
        input_size: 讀取資料的影像大小，意義為 (height, width)

    Returns:
        dictionary of transforms
            train: 針對訓練集的轉換函數
            val: 針對驗證集的轉換函數
    """
    image_map = image_preprocess_map()
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.RandomAffine(
                    degrees=cfg.ROTATE_ANGLE, translate=cfg.TRANSLATION
                ),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=cfg.MEAN_LIST, std=cfg.STD_LIST),
            ]
        ),
        "val": image_map,
    }
    return data_transforms


def get_dataloaders_dict(data_transforms: dict, batch_size: int, data_path: str) -> dict:
    """
    使用資料集轉換函數，取得 train 與 val 的 dataloader
    因為製造資料的方式是用 ImageFolder，所以 data_path 的資料夾就是影像檔的 label 名稱
    且他們在 dataloader 的 index 是按照 label 字母排序

    Args:
        data_transforms: dictionary of transforms
            train: 針對訓練集的轉換函數
            val: 針對驗證集的轉換函數
        batch_size: 這個 dataloader 的每批輸出量
        data_path: 讀取影像檔的路徑

    Returns:
        dictionary of dataloaders
            train: 提取訓練集的 dataloader
            val: 提取驗證集的 dataloader
    """
    # Create training and validation datasets
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_path, x), data_transforms[x])
        for x in ["train", "val"]
    }
    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=my_collate_fn,
        )
        for x in ["train", "val"]
    }
    return dataloaders_dict
