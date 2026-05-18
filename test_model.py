from utils.utils import *
from dataloader.data_loader import *
from models.PixelNet import PixelNet
from configs import DefaultConfigs
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,average_precision_score
import matplotlib.pyplot as plt

# from models.tall_swin import TALL_SWIN
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'



config = DefaultConfigs()
config.val_dir = r"H:\_Data\wang_npr\wang_test"
# config.val_dir = r"H:\_Data\diffusion_datasets\diffusion_datasets"
config.batch_size = 1

def model_test(model, device, test_loader):
    model.eval()
    all_labels = []
    all_scores = []
    all_preds = []

    correct = 0
    with torch.no_grad():
        for freq, target in test_loader:
            freq, target = freq.float().to(device), target.float().to(device)

            outputs = model( freq)

            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(target.view_as(predicted)).sum().item()
            all_labels.extend(target.cpu().numpy())
            all_scores.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())


    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("./results/"+ os.path.basename(config.val_dir) + ".png")
    assert len(all_labels) == len(all_scores), "Lengths of labels and scores must be the same."
    assert any(all_labels), "There must be at least one positive label."

    ap = average_precision_score(all_labels, all_scores)


    print('Test:  accuracy: {}/{} ({:.3f}%),  ap: {},'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), ap))
    return correct / len(test_loader.dataset), ap
def main():
    setup_seed(config.seed)

    transform_test = transforms.Compose([

        transforms.CenterCrop(224, ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_data = ImageDataset(root=config.val_dir,
                            label_map=config.label_map,
                            transform=transform_test,
                            )
    test_loader = DataLoader(dataset=test_data,
                             batch_size=config.batch_size,
                             shuffle=False,
                             num_workers=config.workers,
                             pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    pixelnet = PixelNet(input_channel=config.input_channel, mode=config.mode, kernels=config.kernal, arrs=config.arrs,
                  stride=config.stride, padding=0)


    pixelnet.load_state_dict(torch.load("./results/parameter.pth",map_location="cpu",weights_only=True))
    cfnet = pixelnet.to(device)
    model_test(cfnet, device, test_loader, )





if __name__ == '__main__':
    main()