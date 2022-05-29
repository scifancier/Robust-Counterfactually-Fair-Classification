import numpy as np
import torch
from scipy import stats
from scipy.special import softmax
from math import inf
import random


def get_instance_noisy_label(x, y, noise_rate=0.2, label_num=2, norm_std=0.1):
    P = []
    flip_distribution = stats.truncnorm((0 - noise_rate) / norm_std, (1 - noise_rate) / norm_std, loc=noise_rate,
                                        scale=norm_std)
    flip_rate = flip_distribution.rvs(y.shape[0])
    feature_size = x.shape[1]
    w = np.random.randn(label_num, feature_size, label_num)
    label = y.copy()
    label[label == -1] = 0
    label = label.astype(int)
    for i in range(len(label)):
        b = label[i]
        a = x[i].dot(w[0])
        a[b] = -inf
        a = flip_rate[i] * softmax(a, axis=0)
        a[b] += 1 - flip_rate[i]
        P.append(a)
    l = [-1, 1]
    noisy_label = np.array([np.random.choice(l, p=P[i]) for i in range(label.shape[0])])
    error = y != noisy_label
    true_rate = error.sum() / y.shape[0]
    print('True noise rate:', true_rate)

    return noisy_label


def sample_index(data, label, index):
    test_a = data['a'][index]
    test_z = data['z'][index]
    test_w = data['w'][index]
    test_y = label[index]

    return test_a, test_z, test_w, test_y


# flip clean labels to noisy labels
# train set and val set and test set split
def dataset_split(data, label, noise_rate=0.5, split_per=0.9, random_seed=1, noise_type='none'):
    num_samples = int(label.shape[0])
    index = np.arange(num_samples)
    np.random.seed(random_seed)
    np.random.shuffle(index)
    unit_num = int(num_samples * (1 - split_per))
    print('test data number:', unit_num)

    test_index = index[:unit_num]
    test_a, test_z, test_w, test_y = sample_index(data, label, test_index)

    train_val_index = index[unit_num:]
    tv_a, tv_z, tv_w, tv_y = sample_index(data, label, train_val_index)

    if noise_rate > 0:
        if noise_type == 's':
            corrupted_index = np.random.choice(tv_y.shape[0], int(tv_y.shape[0] * noise_rate), replace=False)
            print(corrupted_index[:10])
            tv_y[corrupted_index] = tv_y[corrupted_index] * (-1)
        if noise_type == 'i':
            x_whole_feature = np.hstack((tv_a.reshape(-1, 1), tv_z, tv_w))
            tv_y = get_instance_noisy_label(x_whole_feature, tv_y, noise_rate, label_num=2, norm_std=0.1)

    num_samples = tv_y.shape[0]
    index = np.arange(num_samples)
    np.random.shuffle(index)
    unit_num = int(num_samples * (1 - split_per))
    print('validation data number:', unit_num)

    # update data to tv adta
    tv_data = {'a': tv_a, 'z': tv_z, 'w': tv_w}

    val_index = index[:unit_num]
    val_a, val_z, val_w, val_y = sample_index(tv_data, tv_y, val_index)

    train_index = index[unit_num:]
    train_a, train_z, train_w, train_y = sample_index(tv_data, tv_y, train_index)

    print('train data number:', train_y.shape[0])
    train_y[train_y==-1] = 0
    val_y[val_y==-1] = 0
    test_y[test_y==-1] = 0

    return train_a, train_z, train_w, train_y, val_a, val_z, val_w, val_y, test_a, test_z, test_w, test_y

def st_get_instance_noisy_label(x, y, noise_rate=0.2, label_num=2, norm_std=0.1):
    P = []
    flip_distribution = stats.truncnorm((0 - noise_rate) / norm_std, (1 - noise_rate) / norm_std, loc=noise_rate,
                                        scale=norm_std)
    flip_rate = flip_distribution.rvs(y.shape[0])
    feature_size = x.shape[1]
    w = np.random.randn(label_num, feature_size, label_num)
    label = y.copy()
    label[label == -1] = 0
    label = label.astype(int)
    for i in range(len(label)):
        b = label[i]
        a = x[i].dot(w[0])
        a[b] = -inf
        a = flip_rate[i] * softmax(a, axis=0)
        a[b] += 1 - flip_rate[i]
        P.append(a)
    l = [-1, 1]
    noisy_label = np.array([np.random.choice(l, p=P[i]) for i in range(label.shape[0])])
    error = y != noisy_label
    true_rate = error.sum() / y.shape[0]
    print('True noise rate:', true_rate)

    return noisy_label, np.array(P)

def st_dataset_split(data, label, noise_rate=0.5, split_per=0.9, random_seed=1, noise_type='none'):
    num_samples = int(label.shape[0])
    index = np.arange(num_samples)
    np.random.seed(random_seed)
    np.random.shuffle(index)
    unit_num = int(num_samples * (1 - split_per))
    print('test data number:', unit_num)

    test_index = index[:unit_num]
    test_a, test_z, test_w, test_y = sample_index(data, label, test_index)

    train_val_index = index[unit_num:]
    tv_a, tv_z, tv_w, tv_y = sample_index(data, label, train_val_index)

    if noise_rate > 0:
        if noise_type == 's':
            corrupted_index = np.random.choice(tv_y.shape[0], int(tv_y.shape[0] * noise_rate), replace=False)
            print(corrupted_index[:10])
            tv_y[corrupted_index] = tv_y[corrupted_index] * (-1)
        if noise_type == 'i':
            x_whole_feature = np.hstack((tv_a.reshape(-1, 1), tv_z, tv_w))
            tv_y, trans = st_get_instance_noisy_label(x_whole_feature, tv_y, noise_rate, label_num=2, norm_std=0.1)

    num_samples = tv_y.shape[0]
    index = np.arange(num_samples)
    np.random.shuffle(index)
    unit_num = int(num_samples * (1 - split_per))
    print('validation data number:', unit_num)

    # update data to tv adta
    tv_data = {'a': tv_a, 'z': tv_z, 'w': tv_w}

    val_index = index[:unit_num]
    val_a, val_z, val_w, val_y = sample_index(tv_data, tv_y, val_index)

    train_index = index[unit_num:]
    train_a, train_z, train_w, train_y = sample_index(tv_data, tv_y, train_index)

    trans = trans[train_index]

    print('train data number:', train_y.shape[0])
    train_y[train_y==-1] = 0
    val_y[val_y==-1] = 0
    test_y[test_y==-1] = 0

    return train_a, train_z, train_w, train_y, val_a, val_z, val_w, val_y, test_a, test_z, test_w, test_y, trans

