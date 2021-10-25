"""
針對資料集算成效運算
"""
import torch


def evaluate(model, dataloader, device) -> float:
    """
    使用 model 針對 dataloader 計算成效

    Args:
        model: 模型物件
        dataloader: torch 的資料集合
        device: 使用運算的機器，cpu or gpu

    Returns:
        計算完的成效，即 model 對 dataloader 的 accuracy
    """
    model.eval()
    correct_count = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct_count += torch.sum(preds == labels.data)
    accuracy = correct_count.double() / len(dataloader.dataset)
    return accuracy
