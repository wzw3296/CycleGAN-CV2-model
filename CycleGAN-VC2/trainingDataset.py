from torch.utils.data.dataset import Dataset
import torch
import numpy as np

class trainingDataset(Dataset):
    def __init__(self, src_data, target_data, n_frames=128):
        self.src_data = src_data
        self.target_data = target_data
        self.n_frames = n_frames

    def __getitem__(self, index):
        src = self.src_data
        target = self.target_data
        n_frames = self.n_frames

        self.length = min(len(src), len(target))
        num_samples = min(len(src), len(target))
        src_data_id = np.arange(len(src))
        target_data_id = np.arange(len(target))
        np.random.shuffle(src_data_id)
        np.random.shuffle(target_data_id)
        src_data_id_subset = src_data_id[:num_samples]
        target_data_id_subset = target_data_id[:num_samples]
        train_src = list()
        train_target = list()

        for src_id, target_id in zip(src_data_id_subset, target_data_id_subset):
            data_src = src[src_id]
            src_total = data_src.shape[1]
            assert src_total >= n_frames
            src_begin = np.random.randint(src_total - n_frames + 1)
            src_end = src_begin + n_frames
            train_src.append(data_src[:, src_begin:src_end])

            data_target = target[target_id]
            target_total = data_target.shape[1]
            assert target_total >= n_frames
            target_begin = np.random.randint(target_total - n_frames + 1)
            target_end = target_begin + n_frames
            train_target.append(data_target[:, target_begin:target_end])

        train_src = np.array(train_src)
        train_target = np.array(train_target)

        return train_src[index], train_target[index]

    def __len__(self):
        return min(len(self.src_data), len(self.target_data))
