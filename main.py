
import torch.optim as optim
from utils.utils import *
from utils.gen_matirx import *
from dataloader.data_loader import *
from models.PixelNet import PixelNet
from configs import DefaultConfigs

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
config = DefaultConfigs()

def main():
    torch.cuda.empty_cache()
    setup_seed(config.seed)

    # filename = f"matrix_.txt"
    # file_path = os.path.join(config.dir, filename)  # 完整路径
    # # 将矩阵保存到指定路径
    # save_matrix_to_txt(config.arrs, file_path)

    transform_train = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.3, 0.6), ratio=(0.3, 1.3), value=0, inplace=False),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = ImageDataset(root=config.train_dir,
                              label_map=config.label_map,
                              transform=transform_train,
                              )

    train_loader = DataLoader(dataset=train_data,
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=config.workers,
                              # pin_memory=True
                              )
    transform_test = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_data = ImageDataset(root=config.val_dir,
                            label_map=config.label_map,
                            transform=transform_test,
                            )
    val_loader = DataLoader(dataset=val_data,
                             batch_size=config.batch_size,
                             shuffle=False,
                             num_workers=config.workers,
                            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    pixelnet = PixelNet(input_channel=config.input_channel, mode=config.mode, kernels=config.kernal, arrs=config.arrs, stride=config.stride, padding=0)

    pixelnet = pixelnet.to(device)

    scaler = GradScaler()

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(pixelnet.parameters(), lr=config.lr, weight_decay=1e-4)


    best_acc = 0.0
    count = 0

    for epoch in range(0, config.epochs ):
        adjust_learning_rate(optimizer, config.lr, epoch)
        loss, acc1 = train(pixelnet, device, train_loader,
                           criterion, optimizer, scaler, epoch)

        val_loss1, val_acc1 = test(pixelnet, device, val_loader, criterion)


        with open(os.path.join(result_dir, 'train.txt'), "a+") as f1:
            f1.write('{}\t{}\t{}\t\n'.format(epoch, round(loss, 3), round(acc1, 4)))
        with open(os.path.join(result_dir, 'val.txt'), "a+") as f2:
            f2.write('{}\t{}\t{}\t\n'.format(epoch, round(val_loss1, 3), round(val_acc1, 4)))

        # 保存效果最好的模型参数
        if val_acc1 > best_acc:
            best_acc = val_acc1
            count = 0
            torch.save(pixelnet.state_dict(), 'results/parameter.pth')
        count = count + 1
        if count > 9 :
            break
    print('验证集的最高准确率为： %.03f' % (best_acc*100))


    model = PixelNet(input_channel=config.input_channel, mode=config.mode, kernels=config.kernal, arrs=config.arrs,
                  stride=config.stride, padding=0)

    model.load_state_dict(torch.load("./results/parameter.pth"))
    model.to(device)
    each_dataset_test(model, config)


if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    # 清空results下所有文件
    result_dir = "results"
    if os.listdir(result_dir):
        for file in os.listdir(result_dir):
            os.remove(os.path.join(result_dir, file))
    main()