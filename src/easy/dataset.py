import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class MusicDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.features = torch.tensor(self.data.values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]

if __name__ == "__main__":
    train_dataset = MusicDataset("C:/Users/user/OneDrive/Documents/musicdata/data/features/easy/train.csv")
    valid_dataset = MusicDataset("C:/Users/user/OneDrive/Documents/musicdata/data/features/easy/valid.csv")
    test_dataset = MusicDataset("C:/Users/user/OneDrive/Documents/musicdata/data/features/easy/test.csv")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print("Train dataset shape:", train_dataset.features.shape)
    print("Valid dataset shape:", valid_dataset.features.shape)
    print("Test dataset shape:", test_dataset.features.shape)



