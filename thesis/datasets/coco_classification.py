from pycocotools.coco import COCO
from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision import transforms

class MSCOCODataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.coco = COCO(annotation_file)
        self.ids = list(self.coco.imgToAnns.keys())
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        labels = [ann['category_id'] for ann in anns]

        path = self.coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.root_dir, path)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Choose the first label for simplicity in classification
        return image, labels[0]
