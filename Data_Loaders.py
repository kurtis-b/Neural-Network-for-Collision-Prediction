import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split


class Nav_Dataset(dataset.Dataset):
    def __init__(self, data):
        # self.data = np.genfromtxt('saved/training_data.csv', delimiter=',')
        self.data = data
        # STUDENTS: it may be helpful for the final part to balance the distribution of your collected data

        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data)  # fits and transforms
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb"))  # save to normalize at inference

    def __len__(self):
        # STUDENTS: __len__() returns the length of the dataset
        return len(self.normalized_data)

    def __getitem__(self, idx):
        # STUDENTS: for this example, __getitem__() must return a dict with entries {'input': x, 'label': y}
        # x and y should both be of type float32. There are many other ways to do this, but to work with autograding
        # please do not deviate from these specifications.
        if not isinstance(idx, int):
            idx = idx.item()
        x = [i for i in self.normalized_data[idx][:6]]
        y = [self.normalized_data[idx][-1]]
        sample = {'input': torch.tensor(x, dtype=torch.float32), 'label': torch.tensor(y, dtype=torch.float32)}
        return sample


class Data_Loaders():
    def __init__(self, batch_size):
        # STUDENTS: randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader
        # make sure your split can handle an arbitrary number of samples in the dataset as this may vary
        # Load and preprocess your data
        data_input = np.genfromtxt('saved/training_data.csv', delimiter=',')
        self.data = sorted(data_input, key=lambda l: l[6])  # Sort by the last element
        # Find the first occurrence of a collision
        first_collision = 0
        for i, row in enumerate(self.data):
            if row[6] == 1:
                first_collision = i
                break

        # Split data into training and testing sets
        train_data_no_collisions, test_data_no_collisions = train_test_split(self.data[:first_collision], test_size=0.2)
        train_data_collisions, test_data_collisions = train_test_split(self.data[first_collision:], test_size=0.2)
        self.train_loader = data.DataLoader(Nav_Dataset(train_data_no_collisions + train_data_collisions), batch_size,
                                            shuffle=True)
        self.test_loader = data.DataLoader(Nav_Dataset(test_data_no_collisions + test_data_collisions), batch_size,
                                           shuffle=True)


def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    # STUDENTS : note this is how the dataloaders will be iterated over, and cannot be deviated from
    print("train_loader")
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
        print('sample:', sample['input'], 'label:', sample['label'])
        print('sample:', sample['input'].shape, 'label:', sample['label'].shape)
    print("test_loader")
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']
        print('sample:', sample['input'], 'label:', sample['label'])


if __name__ == '__main__':
    main()
