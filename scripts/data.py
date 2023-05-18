import torch
from torch.utils.data import Dataset
from PIL import Image

class FacialKeypointsDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = Image.fromarray(self.dataframe.iloc[idx, -1])
        keypoints = self.dataframe.iloc[idx, :-1].values.astype('float')

        if self.transform:
            image = self.transform(image)

        return image, keypoints