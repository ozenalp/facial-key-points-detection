import numpy as np
import torch
from torch.utils.data import Dataset

class FacialKeypointsDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_string = self.dataframe.iloc[idx, -1]
        image = np.array([int(item) for item in image_string.split()]).reshape(96, 96, 1)
        image = image / 255.
        keypoints = self.dataframe.iloc[idx, :-1].values.astype('float32')

        sample = {'image': image, 'keypoints': keypoints}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1)).copy()  # Added .copy()

        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(keypoints)}



class CustomHorizontalFlip:
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        image = image[:, ::-1, :]
        keypoints = keypoints.copy()
        keypoints[::2] = 96. - keypoints[::2]  # Assuming image size is 96x96 and keypoints are scaled accordingly
        return {'image': image, 'keypoints': keypoints}


class FlipVertical:
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        image = image[::-1, :, :]
        keypoints = keypoints.copy()
        keypoints[1::2] = 96. - keypoints[1::2]  # Assuming image size is 96x96 and keypoints are scaled accordingly
        return {'image': image, 'keypoints': keypoints}


class IncreaseBrightness:
    def __init__(self, value):
        self.value = value

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        image = np.clip(image + self.value, 0., 1.)  # Ensure values are within [0, 1]
        return {'image': image, 'keypoints': keypoints}


class DecreaseBrightness:
    def __init__(self, value):
        self.value = value

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        image = np.clip(image - self.value, 0., 1.)  # Ensure values are within [0, 1]
        return {'image': image, 'keypoints': keypoints}


