import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset

class ImageDataset(Dataset):
    def __init__(self, root, label_map,transform):
        # dataset = torchvision.datasets.ImageFolder(root, transform)
        # self.dataset = dataset
        self.img_path = []
        self.label_map = label_map
        self.label_map = {'real': 0, 'fake': 1} if label_map is None else label_map
        self.labels = []
        self.transform = transform
        for subdir, _, files in os.walk(root):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    path = os.path.join(subdir, file)
                    self.img_path.append(path)
                    self.labels.append(self.get_label_from_path(path))
    def get_label_from_path(self, path):
        """
        Determine the label of an image based on its file path.
        """
        for key in self.label_map:
            if key.lower() in path.lower():
                return self.label_map[key]
        # Return a default label if no match found
        return -1
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        # 加载图像
        # CHW
        img = Image.open(self.img_path[idx]).convert('YCbCr')  # 转换为YCbCr颜色空间
        if self.transform:
            img = self.transform(img)

        # 获取标签
        label = self.labels[idx]

        return img, label
