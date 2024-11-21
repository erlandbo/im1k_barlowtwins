from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder



class ImageNetIndex(Dataset):
    def __init__(self, root, transform):
        self.imagefolder = ImageFolder(root, transform=transform)

    def __len__(self):
        return self.imagefolder.__len__()

    def __getitem__(self, index):
        images, targets = self.imagefolder.__getitem__(index)
        return images, targets, index

