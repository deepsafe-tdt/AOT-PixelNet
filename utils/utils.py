from torch.utils.data import DataLoader
import os
import torchvision.transforms as transforms
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast,GradScaler
from sklearn.metrics import roc_auc_score, accuracy_score,average_precision_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataloader.data_loader import ImageDataset





# 定义一个可以设置随机种子的函数
def setup_seed(seed):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 设置学习率
def adjust_learning_rate(optimizer, learning_rate, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 20))
    if lr < 0.00001:
        lr = 0.00001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# 训练网络
def train(model, device, train_loader,
            criterion, optimizer,
            scaler, epoch, centerloss=None, center_optimizer = None):
    model.train()

    trained_samples = 0
    sum_loss = 0.0
    train_loss = 0
    total = 0.0
    correct = 0.0
    # pbar = tqdm(train_loader, desc=f'Train Epoch {epoch}', unit='batch')
    # for batch_idx, data in enumerate(train_loader):
    with tqdm(train_loader, desc=f'Train Epoch {epoch}', unit='batch',leave=False ) as t:
        for data in t:
            freq, target = data
            freq, target = freq.float().to(device), target.long().to(device) #celoss需要longtensor
    
            optimizer.zero_grad()

            outputs = model(freq)
            loss = criterion(outputs.squeeze(), target)
            loss.backward()
            optimizer.step()
    

            total += target.size(0)

            sum_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == target.data).item()
            epoch_loss = sum_loss / total
            epoch_acc = 100 * correct / total
            t.set_postfix({'Loss': f'{epoch_loss:.4f}', 'Accuracy': f'{epoch_acc:.4f}'})


    print(f'Train Epoch: {epoch}\tAverage Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}',flush=True)

    print('\r')

    return epoch_loss, epoch_acc



# 测试模型
def test(model, device, test_loader, criterion, centerloss=None):
    model.eval()
    all_labels = []
    all_scores = []
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        with tqdm(test_loader, desc='Test', unit='batch',leave=False) as t:
            for freq, target in t:
                freq, target = freq.float().to(device), target.long().to(device)

                outputs = model(freq)
                # loss = criterion(outputs.squeeze(), target)
                loss = criterion(outputs, target)
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(target.view_as(predicted)).sum().item()

                all_labels.extend(target.cpu().numpy())
                all_scores.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())


    ap = average_precision_score(all_labels, all_scores)
    test_loss /= len(test_loader.dataset)

    test_acc = correct / len(test_loader.dataset)
    print(f'\nTest set: Accuracy: {test_acc:.4f}, AP: {ap:.4f}\n',flush=True)
    return test_loss, test_acc

def model_test(model, test_loader, device):
    model.eval()
    all_labels = []
    all_scores = []
    all_preds = []

    correct = 0
    with torch.no_grad():
        for freq, target in test_loader:
            freq, target = freq.float().to(device), target.float().to(device)

            outputs = model(freq)

            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(target.view_as(predicted)).sum().item()
            all_labels.extend(target.cpu().numpy())
            all_scores.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())


    assert len(all_labels) == len(all_scores), "Lengths of labels and scores must be the same."

    ap = average_precision_score(all_labels, all_scores)


    return correct / len(test_loader.dataset), ap

def each_dataset_test(model, config):

    results_file = os.path.join("./results", os.path.basename(config.test_dir) + ".txt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model_names = [d for d in os.listdir(config.test_dir) if os.path.isdir(os.path.join(config.test_dir, d))]

    accs = {}
    aps = {}
    overall_accuracy = 0.0
    overall_ap = 0.0

    for model_name in model_names:
        model_dir = os.path.join(config.test_dir, model_name)
        test_transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = ImageDataset(root=model_dir,
                                 label_map=config.label_map,
                                 transform=test_transform,
                                 )
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        accuracy, ap = model_test(model, dataloader, device)
        accs[model_name] = accuracy
        aps[model_name] = ap
        overall_accuracy += accuracy
        overall_ap += ap
        print(f'Model: {model_name}, Accuracy: {accuracy:.4f}, AP: {ap:.4f}')

    overall_accuracy /= len(model_names)
    overall_ap /= len(model_names)
    print(f'Overall Average Accuracy: {overall_accuracy:.4f}, Overall AP: {overall_ap:.4f}')

    # 保存结果到文件
    with open(results_file, 'w') as f:
        for model_name in model_names:
            f.write(f'Model: {model_name}, Accuracy: {accs[model_name]:.4f}, AP: {aps[model_name]:.4f}\n')
        f.write(f'Overall Average Accuracy: {overall_accuracy:.4f}\n')
        f.write(f'Overall Average AP: {overall_ap:.4f}\n')
