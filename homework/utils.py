import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import pandas as pd
import numpy as np
from torchvision.transforms import functional as F
#from homework.dense_transforms import ToTensor, Compose, RandomHorizontalFlip, label_to_pil_image
#from homework import dense_transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']
DENSE_LABEL_NAMES = ['background', 'kart', 'track', 'bomb/projectile', 'pickup/nitro']
# Distribution of classes on dense training set (background and track dominate (96.0%)
DENSE_CLASS_DISTRIBUTION = [0.52683655, 0.02929112, 0.4352989, 0.0044619, 0.00411153]


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path, transform, args):
        self.path = dataset_path
        self.transform = transform
        self.args = args



        # Images
        image_files = pd.read_csv(os.path.join(dataset_path, 'labels.csv'))['file'].tolist()
        self.image_paths = [os.path.join(dataset_path, x) for x in image_files]

        # Labels
        self.str_labels = pd.read_csv(os.path.join(dataset_path, 'labels.csv'))['label'].tolist()
        label_dict = {"background": 0, "kart": 1, "pickup": 2, "nitro": 3, "bomb": 4, "projectile": 5}
        self.int_labels = [label_dict[x] for x in self.str_labels]

    def __len__(self):
        return len(self.int_labels)

    def __getitem__(self, idx):
        label = self.int_labels[idx]

        # Convert filepath to tensor
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(self.args, image)
        else:
            totensor = ToTensor()
            image = totensor(image)

        return image, label


class Transform:
    def __call__(self, args, image):

        # Random Cropping
        random_crop = None
        if args.rand_crop is not None:
            random_crop = transforms.RandomCrop(args.rand_crop)

        # Horizontal flipping
        h_flip = None
        if args.h_flip is not None:
            h_flip = transforms.RandomHorizontalFlip(args.h_flip)

        # Vertical flipping
        v_flip = None
        if args.v_flip is not None:
            v_flip = transforms.RandomVerticalFlip(args.v_flip)

        # Random Rotation
        rand_rotate = None
        if args.rand_rotate is not None:
            rand_rotate = transforms.RandomRotation(args.rand_rotate)

        # Brightness
        brightness = None
        if args.brightness is not None:
            brightness = transforms.ColorJitter(brightness=args.brightness)

        # Contrast
        contrast = None
        if args.contrast is not None:
            contrast = transforms.ColorJitter(contrast=args.contrast)

        # Saturation
        saturation = None
        if args.saturation is not None:
            saturation = transforms.ColorJitter(saturation=args.saturation)

        # Hue
        hue = None
        if args.hue is not None:
            hue = transforms.ColorJitter(hue=args.hue)

        transformations = [random_crop, h_flip, v_flip, rand_rotate, brightness, contrast, saturation, hue]
        transformations = [x for x in transformations if x is not None]
        transformations.append(transforms.ToTensor())
        transform = transforms.Compose(transformations)
        return transform(image)


class ToTensor:
    def __call__(self, image):
        transform = transforms.Compose([transforms.ToTensor()])
        return transform(image)


class DenseSuperTuxDataset(Dataset):
    def __init__(self, dataset_path, transform):
        from glob import glob
        from os import path
        self.files = []
        for im_f in glob(path.join(dataset_path, '*_im.jpg')):
            self.files.append(im_f.replace('_im.jpg', ''))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        b = self.files[idx]
        im = Image.open(b + '_im.jpg')
        lbl = Image.open(b + '_seg.png')
        if self.transform is not None:
            im = self.transform(im)
            shape = np.shape(np.array(lbl))
            lbl = np.array(lbl).flatten()
            lbl = [int(x) for x in lbl]
            lbl = np.reshape(lbl, shape)
            lbl = self.transform(lbl)

        return im, lbl


def load_data(dataset_path, args, transform, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path=dataset_path, args=args, transform=transform)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def load_dense_data(dataset_path, transform, num_workers=0, batch_size=32):
    dataset = DenseSuperTuxDataset(dataset_path, transform)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def _one_hot(x, n):
    return (x.view(-1, 1) == torch.arange(n, dtype=x.dtype, device=x.device)).int()


class ConfusionMatrix(object):
    def _make(self, preds, labels):
        label_range = torch.arange(self.size, device=preds.device)[None, :]
        preds_one_hot, labels_one_hot = _one_hot(preds, self.size), _one_hot(labels, self.size)
        return (labels_one_hot[:, :, None] * preds_one_hot[:, None, :]).sum(dim=0).detach()

    def __init__(self, size=5):
        """
        This class builds and updates a confusion matrix.
        :param size: the number of classes to consider
        """
        self.matrix = torch.zeros(size, size)
        self.size = size

    def add(self, preds, labels):
        """
        Updates the confusion matrix using predictions `preds` (e.g. logit.argmax(1)) and ground truth `labels`
        """
        self.matrix = self.matrix.to(preds.device)
        self.matrix += self._make(preds, labels).float()

    @property
    def class_iou(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(0) + self.matrix.sum(1) - true_pos + 1e-5)

    @property
    def iou(self):
        return self.class_iou.mean()

    @property
    def global_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos.sum() / (self.matrix.sum() + 1e-5)

    @property
    def class_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(1) + 1e-5)

    @property
    def average_accuracy(self):
        return self.class_accuracy.mean()

    @property
    def per_class(self):
        return self.matrix / (self.matrix.sum(1, keepdims=True) + 1e-5)


if __name__ == '__main__':
    dataset = DenseSuperTuxDataset('dense_data/train', transform=Compose(
        [RandomHorizontalFlip(), ToTensor()]))
    from pylab import show, imshow, subplot, axis

    for i in range(15):
        im, lbl = dataset[i]
        subplot(5, 6, 2 * i + 1)
        imshow(F.to_pil_image(im))
        axis('off')
        subplot(5, 6, 2 * i + 2)
        imshow(label_to_pil_image(lbl))
        axis('off')
    show()
    import numpy as np

    c = np.zeros(5)
    for im, lbl in dataset:
        c += np.bincount(lbl.view(-1), minlength=len(DENSE_LABEL_NAMES))
    print(100 * c / np.sum(c))
