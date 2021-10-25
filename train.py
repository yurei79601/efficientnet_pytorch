"""
training process of efficientnet
"""
import os
import time
from copy import deepcopy
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from utils import check_path_exists
from data_manager import get_data_transforms, get_dataloaders_dict
from evaluation import evaluate
import config as cfg


def save_model_weight(model, model_path, iteration):
    """save input model weight"""
    save_model_path = os.path.join(
        model_path, f"efficientnet_{iteration:04}.pth"
    )
    torch.save(model.state_dict(), save_model_path)
    print(f"save model at {save_model_path}")


def get_parameters_to_update(model, feature_extract):
    """
    提取欲更新的模型參數

    Args:
        model: 模型物件
        feature_extract: 是否提取特徵
            - True 為特徵提取
            - False 為微調

    Returns:
        model 可以被更新的參數
    """
    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print("\t", name)
    return params_to_update


def initialize_model(device, num_classes):
    """
    模型物件化並且修改 fully connected 的輸出曾數量

    Args:
        device: 使用運算的機器，cpu or gpu
        num_classes: 分類的數量

    Returns:
        將新的分類模型實體化
    """
    model = EfficientNet.from_pretrained(cfg.MODEL_NAME)
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, num_classes)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    return model


def train_one_epoch(
    model, dataloader, criterion, optimizer, device, is_inception=False
):
    """
    針對 dataloader 裡面的資料訓練一次

    Args:
        model: 訓練階段的模型物件
        dataloaders: 訓練資料的 dataloader
        criterion: loss function，計算 loss 的方式
        optimizer: 更新參數的方式
        device: 使用運算的機器，cpu or gpu
        is_inception: 是否為 inception Net，計算 loss 的方式會不同

    Returns:
        model: 訓練完畢的 model 物件
        epoch_loss: 本次模型針對訓練集的 loss
        epoch_accuracy: 本次模型針對訓練集的 accuracy
    """
    running_loss, running_corrects = 0, 0
    # Set model to training mode
    model.train()
    # Iterate over data.
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        with torch.set_grad_enabled(True):
            if is_inception:
                outputs, aux_outputs = model(inputs)
                loss1 = criterion(outputs, labels)
                loss2 = criterion(aux_outputs, labels)
                loss = loss1 + 0.4 * loss2
            else:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = running_corrects / len(dataloader.dataset)
    return model, epoch_loss, epoch_accuracy


def train_model(
    model,
    dataloaders: dict,
    criterion,
    optimizer,
    device,
    num_epochs=25,
    is_inception=False,
):
    """
    針對 dataloaders 裡面的 train 以及 val 重複地做訓練以及驗證。共做 num_epochs 次。

    Args:
        model: 初始模型架構
        dataloaders: 訓練資料的 dataloader，分為 train, val 兩個物件
        criterion: loss function，計算 loss 的方式
        optimizer: 更新參數的方式
        device: 使用運算的機器，cpu or gpu
        num_epoch: 訓練次數
        is_inception: 是否為 inception Net，計算 loss 的方式會不同

    Returns:
        model: 訓練完畢的 model 物件
        val_acc_history (list): 訓練過程中每個 epoch 對 validation data 的 accuracy
    """
    since = time.time()

    val_acc_history = []

    best_model_wts = deepcopy(model.state_dict())
    best_accuracy = 0.0

    for epoch in range(1, num_epochs + 1):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print("-" * 10)

        # Each epoch has a training and validation phase
        model, train_loss, train_accuracy = train_one_epoch(
            model,
            dataloaders["train"],
            criterion,
            optimizer,
            device,
            is_inception,
        )
        valid_accuracy = evaluate(model, dataloaders["val"], device)

        print(
            "Epoch: {} Training Loss: {:.4f} Training Acc: {:.4f} Valid Acc: {:.4f}".format(
                epoch, train_loss, train_accuracy, valid_accuracy
            )
        )

        # deep copy the model
        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            best_model_wts = deepcopy(model.state_dict())
            save_model_weight(model, cfg.MODEL_SAVE_PATH, epoch)

        val_acc_history.append(float(valid_accuracy))

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_accuracy))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def main(device):
    """
    根據 device 去跑整個訓練過程
    訓練完畢將產出
        - pytorch 模型權重檔，格式為 .pth
        - 一個 pickle 檔，記錄訓練過程中的 loss
    """
    model = initialize_model(device, cfg.NUMBER_CLASSES)

    params_to_update = get_parameters_to_update(model, cfg.FEATURE_EXTACT)
    optimizer = optim.Adam(
        params_to_update, lr=cfg.LR, betas=(0.9, 0.999), eps=1e-9
    )
    data_transforms = get_data_transforms(cfg.INPUT_SIZE)
    dataloaders_dict = get_dataloaders_dict(data_transforms, cfg.BATCH_SIZE, cfg.DATA_PATH)
    # Setup the loss function
    criterion = nn.CrossEntropyLoss()
    # Train and evaluate
    final_model, histotry_loss = train_model(
        model,
        dataloaders_dict,
        criterion,
        optimizer,
        device,
        num_epochs=cfg.EPOCH,
        is_inception=(cfg.MODEL_NAME == "inception"),
    )
    torch.save(
        final_model.state_dict(),
        os.path.join(cfg.MODEL_SAVE_PATH, cfg.MODEL_SAVE_NAME),
    )
    loss_df = pd.DataFrame(histotry_loss, columns=['loss'])
    loss_df.to_pickle(os.path.join(cfg.MODEL_SAVE_PATH, "loss.pkl"))


if __name__ == "__main__":
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    check_path_exists(cfg.MODEL_SAVE_PATH)
    main(DEVICE)
