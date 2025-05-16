from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class NumpyToPilImage(object):
    def __call__(self, image):
        return Image.fromarray(image.astype('uint8'))

class preprocess_video_dataset(Dataset):
    def __init__(self, video, config):
        self.transform = transforms.Compose([
            NumpyToPilImage(),
            transforms.Resize(config["image_size"]),
            transforms.CenterCrop(config["crop_size"]),
            transforms.ToTensor(),
            transforms.Normalize(config["mean"], config["std"])
        ])
        # NCHW
        self.data_list = video

    def __getitem__(self, idx):
        image = self.data_list[idx]
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.data_list)


class PILImageDataset(Dataset):
    def __init__(self, pil_images, transform=None):
        self.pil_images = pil_images
        self.transform = transform

    def __len__(self):
        return len(self.pil_images)

    def __getitem__(self, idx):
        image = self.pil_images[idx]
        if self.transform:
            image = self.transform(image)
        return image