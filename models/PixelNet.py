import torch
import torch.nn.functional as F
import torchvision.models
from torch import nn
import torch_dct as dct
import pytorch_wavelets as wavelets
from models.resnet import resnet50

def compute_zigzag_indices(matrix_size):
    """
    计算指定矩阵大小的 Zigzag 顺序索引。

    Args:
        matrix_size (int): 矩阵的边长，例如 4 表示 4x4 矩阵。

    Returns:
        torch.Tensor: 预计算的 Zigzag 索引，形状为 (num_elements, 2)。
    """
    indices = []
    for sum_idx in range(2 * matrix_size - 1):  # 遍历对角线
        if sum_idx % 2 == 0:
            # 偶数对角线：从左下到右上
            for i in range(min(sum_idx, matrix_size - 1), max(sum_idx - matrix_size, -1), -1):
                indices.append((i, sum_idx - i))
        else:
            # 奇数对角线：从右上到左下
            for j in range(min(sum_idx, matrix_size - 1), max(sum_idx - matrix_size, -1), -1):
                indices.append((sum_idx - j, j))
    return torch.tensor(indices)


def zigzag(input_tensor):
    """
    将张量按照 Zigzag 顺序排列。

    Args:
        input_tensor (torch.Tensor): 输入张量，形状为 (B, C, H, W, block_size, block_size)。
        block_size (int): 每个块的大小，默认是 4 表示 4x4。

    Returns:
        torch.Tensor: Zigzag 排列后的张量，形状为 (B, C, H, W, block_size * block_size)。
    """

    if input_tensor.shape[-2] != input_tensor.shape[-1]:
        raise ValueError(f"输入张量的最后两个维度必须相等。")
    block_size = input_tensor.shape[-2]
    # 获取 Zigzag 索引
    zigzag_indices = compute_zigzag_indices(block_size)
    row_idx, col_idx = zigzag_indices[:, 0], zigzag_indices[:, 1]

    # 按照 Zigzag 索引重新排列
    zigzag_tensor = input_tensor[..., row_idx, col_idx].reshape(
        *input_tensor.shape[:-2], -1
    )

    return zigzag_tensor



class PixelNet(nn.Module):
    def __init__(self,input_channel, mode, kernels, arrs, stride, padding, num_classes=2):
        super(PixelNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.kernels = kernels  # 核大小
        self.arrs = arrs.to(self.device)  # 变换矩阵
        self.stride = stride
        self.padding = padding
        self.resnet = resnet50()
        self.resnet.conv1 = nn.Conv2d(input_channel, 64, kernel_size=(1, 1), stride=1, padding=0, bias=False)
        self.resnet.fc1 = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.AOT_transform(x)
        x = self.resnet(x)
        return x

    def AOT_transform(self,img_batch):
        # print(img_batch.shape)
        """
               img_batch: 形状为 [B, C, H, W]
               返回: 转换后的图像张量，形状为 [B, C*L, H_out, W_out]
               """

        B, C, H, W = img_batch.shape
        kh, kw = self.kernels.shape
        padding = self.padding

        # 对输入图像进行 padding
        img_batch = F.pad(img_batch, (padding, padding, padding, padding))
        # 使用 unfold 提取特征块

        patches = img_batch.unfold(3, kh, self.stride).unfold(2, kw, self.stride).permute(0, 1, 2, 3, 5, 4)

        # 获取输出高度和宽度
        out_h, out_w = patches.shape[2:4]

        if self.mode == "RGB":
            # RGB 模式的归一化

            outputs = patches.flatten(start_dim=-2)


        elif self.mode == "FFT":
            # 使用 torch 计算 DCT
            """"ortho" - 按1 / sqrt(n)归一化（使FFT正交）"""
            patches = torch.fft.fft2(patches, norm="ortho")
            patches = torch.log(torch.abs(patches) + 1) * 20

            outputs = zigzag(patches)

        elif self.mode == "DCT":

            # 使用 torch 计算 DCT
            patches = dct.dct_2d(patches)
            patches = torch.log(torch.abs(patches) + 1) * 20
            outputs = zigzag(patches)

        elif self.mode == "DWT":
            dwt = wavelets.DWTForward(J=1, wave='haar', ).to(self.device)

            ll, high = dwt(patches.reshape(B, -1, kh, kw))
            lh, hl, hh = high[0][:, :, 0, :, :], high[0][:, :, 1, :, :], high[0][:, :, 2, :, :]

            outputs = torch.cat([ll.flatten(start_dim=-2), lh.flatten(start_dim=-2), hl.flatten(start_dim=-2),
                                 hh.flatten(start_dim=-2)], dim=-1)
            outputs = torch.log(torch.abs(outputs) + 1) * 20
            outputs = outputs.view(B, C, out_h, out_w, kw, kh)


            outputs = outputs.flatten(start_dim=-2, end_dim=-1)

        else:
            # 使用 arrs 矩阵转换

            patches = torch.einsum('ij,bchwjk->bchwik', self.arrs, patches)
            patches = torch.einsum('bchwki,ij->bchwkj', patches, self.arrs.t())
            patches = 20 * torch.log(torch.abs(patches) + 1)

            outputs = patches.flatten(start_dim=-2)

        # 调整维度为期望的输出格式[b,ct,h,w]
        outputs = outputs.permute(0, 1, 4, 2, 3).reshape(B, C * kh * kw, out_h, out_w)
        return outputs

if __name__ == '__main__':
    from torchinfo import summary
    import torch
    model = PixelNet(input_channel=192,mode="WT", kernels=torch.randn(8, 8), arrs=torch.eye(8), stride=8, padding=0)

    summary(model,[1,3,224,224], device="cuda")
    import torch
    from thop import profile

    # 创建一个模拟输入张量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # 使用 thop.profile 计算 FLOPs
    flops, params = profile(model, inputs=(dummy_input,))

    print(f"FLOPs: {flops / 1000000000:,}")
    print(f"Number of parameters: {params:,}")