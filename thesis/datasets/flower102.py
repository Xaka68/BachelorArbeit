from torchvision.datasets import Oxford102Flowers
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import image

class Flower102Dataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.dataset = Oxford102Flowers(root=root, split=split, download=True)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
