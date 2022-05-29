import numpy as np
import torch.utils.data as Data
import torch
import tools


class Dataset(Data.Dataset):
    def __init__(self, dataset='adult', train=True, val=False, test=False, dir='', noise_rate=0.5,
                 split_per=0.9, random_seed=1, noise_type='none'):

        self.dataset = dataset
        self.train = train
        self.val = val
        self.test = test
        a = np.load(dir + dataset + '_a.npy')
        z = np.load(dir + dataset + '_z.npy')
        w = np.load(dir + dataset + '_w.npy')
        y = np.load(dir + dataset + '_y.npy')
        data = {'a': a, 'z': z, 'w': w}

        self.train_a, self.train_z, self.train_w, self.train_y, self.val_a, self.val_z, self.val_w, self.val_y, \
            self.test_a, self.test_z, self.test_w, self.test_y = \
            tools.dataset_split(data, y, noise_rate, split_per, random_seed, noise_type)
        self.train_az = np.hstack((self.train_a.reshape(-1, 1), self.train_z))
        self.val_az = np.hstack((self.val_a.reshape(-1, 1), self.val_z))
        self.test_az = np.hstack((self.test_a.reshape(-1, 1), self.test_z))
        

        pass

    def __getitem__(self, index):

        if self.train:
            az, z, w, y = self.train_az[index], self.train_z[index], self.train_w[index], self.train_y[index]
        elif self.val:
            az, z, w, y = self.val_az[index], self.val_z[index], self.val_w[index], self.val_y[index]
        elif self.test:
            az, z, w, y = self.test_az[index], self.test_z[index], self.test_w[index], self.test_y[index]

        az = torch.from_numpy(az)
        z = torch.from_numpy(z)
        w = torch.from_numpy(w)
        y = torch.from_numpy(np.array(y)).long()
        

        return az, z, w, y

    def __len__(self):

        if self.train:
            return len(self.train_y)
        elif self.val:
            return len(self.val_y)
        elif self.test:
            return len(self.test_y)


class st_Dataset(Data.Dataset):
    def __init__(self, dataset='adult', train=True, val=False, test=False, dir='', noise_rate=0.5,
                 split_per=0.9, random_seed=1, noise_type='none'):

        self.dataset = dataset
        self.train = train
        self.val = val
        self.test = test
        a = np.load(dir + dataset + '_a.npy')
        z = np.load(dir + dataset + '_z.npy')
        w = np.load(dir + dataset + '_w.npy')
        y = np.load(dir + dataset + '_y.npy')
        data = {'a': a, 'z': z, 'w': w}

        self.train_a, self.train_z, self.train_w, self.train_y, self.val_a, self.val_z, self.val_w, self.val_y, \
            self.test_a, self.test_z, self.test_w, self.test_y, self.trans = \
            tools.st_dataset_split(data, y, noise_rate, split_per, random_seed, noise_type)
        self.train_az = np.hstack((self.train_a.reshape(-1, 1), self.train_z))
        self.val_az = np.hstack((self.val_a.reshape(-1, 1), self.val_z))
        self.test_az = np.hstack((self.test_a.reshape(-1, 1), self.test_z))

        pass

    def __getitem__(self, index):

        if self.train:
            a, z, w, y, t = self.train_a[index], self.train_z[index], self.train_w[index], self.train_y[index], self.trans[index]
        elif self.val:
            a, z, w, y = self.val_a[index], self.val_z[index], self.val_w[index], self.val_y[index]
        elif self.test:
            a, z, w, y = self.test_a[index], self.test_z[index], self.test_w[index], self.test_y[index]

        a = torch.from_numpy(np.array(a))
        z = torch.from_numpy(z)
        w = torch.from_numpy(w)
        y = torch.from_numpy(np.array(y)).long()

        if self.train:
            return a, z, w, y, t
        else:
            return a, z, w, y

    def __len__(self):

        if self.train:
            return len(self.train_y)
        elif self.val:
            return len(self.val_y)
        elif self.test:
            return len(self.test_y)

