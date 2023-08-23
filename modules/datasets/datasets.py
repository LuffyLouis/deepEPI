# import os
#
# import h5py
import os.path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class EPIDatasets(Dataset):
    def __init__(self, data=None, label=None, cache_file_path=None, transform=None):
        self.label_key = "y"
        self.data_key = "X"
        self.h5 = False
        self.cache_file_path = cache_file_path
        if self.cache_file_path:
            suffix = os.path.basename(self.cache_file_path).split(".")[-1]
            if suffix == "h5":
                self.h5 = True
                # self.data = np.array(self.read_h5(self.data_key, 0))
                with h5py.File(self.cache_file_path, "r") as handle:
                    self.label = torch.tensor(handle[self.label_key], dtype=torch.long)
                pass
                # self.data, self.label = torch.load(self.cache_file_path)
            else:
                self.data, self.label = torch.load(self.cache_file_path)
        else:
            self.data = torch.tensor(data, dtype=torch.float32)
            self.label = torch.tensor(label, dtype=torch.long)
        self.transform = transform
        # with h5py.File(self.h5_file, "r") as handle:
        #     self.keys = list(handle.keys())

    def read_h5(self, dataset_name, idx=None):
        with h5py.File(self.cache_file_path, "r") as handle:
            # self.keys = list(h5_file.keys())
            # for dataset_name in handle:
                # if verbose:
                #     fprint("LOG", "Combing the {}-th file".format(index))
                # if dataset_name ==
            if idx:
                dataset = handle[dataset_name][idx]
                # return np.array(dataset)
                # print(dataset.shape)
                # if len(dataset.shape) == 2:
                #
                #     return torch.tensor(np.expand_dims(dataset,axis=0))
                # else:
                return torch.tensor(dataset)
            else:
                dataset = handle[dataset_name][0]
                # print(dataset.shape)
                return torch.tensor(dataset)

    # def __init(self, *args, **kargs):
    #     pass

    def __len__(self):
        return len(self.label)
        # pass

    def __getitem__(self, idx):
        # with h5py.File(self.h5_file, "r") as handle:
        #     data = handle[self.keys[idx]]
        # 在这里对读取的数据进行必要的预处理，如果有的话
        if self.cache_file_path:
            data = self.read_h5(self.data_key, idx).clone().detach().type(torch.float32)
            label = self.read_h5(self.label_key, idx).clone().detach().type(torch.long)
        else:
            data = self.data[idx]
            label = self.label[idx]
        if self.transform is not None:
            data = self.transform(self.data[idx])

        return data, label

        # pass


if __name__ == '__main__':
    epi_datasets = EPIDatasets("../../output/final_interactions_dna2vec.h5")
    print(epi_datasets.keys)
    print(epi_datasets[0])
    print(len(epi_datasets))