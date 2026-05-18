import torch
import torch.nn as nn
from utils.gen_matirx import gen_matrix

class DefaultConfigs(object):

    seed = 0
    train_dir = r"H:\_Data\wang_npr\train_progan"
    val_dir = r"H:\_Data\wang_npr\wang_test"
    test_dir = r"H:\_Data\wang_npr\wang_test"

    epochs = 50
    batch_size = 32
    workers = 16
    momentum = 0.9
    weight_decay = 1e-4
    lr = 0.0005



    """
     [ "normal", "uniform", "orthogonal", "binary","TOE","random"]
     ["WT","HT","FFT","ST","DCT","DWT"]
    """
    mode = ""
    kernal = torch.randn(8, 8)  # 核的大小
    stride = 8
    pretrained = True
    arrs = torch.tensor(gen_matrix(mode, len(kernal)), dtype=torch.float)
    num_class = 2
    label_map = {'real': 0, 'fake': 1} #标签
    change_channel = False
    input_channel = len(kernal) * len(kernal) * 3 #输入矩阵的通道数
    dir = "./results"
