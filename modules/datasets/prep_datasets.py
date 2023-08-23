import os

import h5py
import numpy as np
import torch
# from torchvision import datasets

from modules import fprint
from sklearn.model_selection import train_test_split

from modules.datasets.datasets import EPIDatasets
from modules.utils.memory import MemoryUseReporter


class PrepDatasets:
    def __init__(self, input_h5_file, key, test_size, split_mode,chunk_size, output_format, output_prefix,compression, random_state=0, verbose=False):
        self.output_prefix = output_prefix
        self.compression = compression
        self.input_h5_file = input_h5_file
        self.chunk_size = chunk_size
        self.split_mode = split_mode
        self.split_ratio = [int(i) for i in self.get_split_ratio(self.split_mode)]
        self.test_size = test_size
        self.random_state = random_state
        self.key = key
        self.format = output_format
        self.verbose = verbose
        self.data = None
        self.label = None
        self.memory_reporter = MemoryUseReporter(os.getpid())
        with h5py.File(self.input_h5_file, "r") as handle:
            self.keys = list(handle.keys())
        if self.key not in self.keys:
            print(self.keys)
            fprint("ERROR","There is not {} key in input: {}".format(self.key,self.input_h5_file))
            raise Exception

        # pass

    # def read_h5(self, key):
    #     with h5py.File(self.input_h5_file, "r") as handle:
    #         for dataset_name in handle:
    #             if dataset_name == key:
    #                 self.data = np.array(handle[dataset_name])
    #                 self.label = np.array(handle[dataset_name + "_label"])

    def read_h5_chunk(self, dataset_name, idx):
        with h5py.File(self.input_h5_file, "r") as handle:
            # self.keys = list(h5_file.keys())
            # for dataset_name in handle:
                # if verbose:
                #     fprint("LOG", "Combing the {}-th file".format(index))
                # if dataset_name ==
            dataset = np.array(handle[dataset_name][idx])
            return dataset

        # pass
    def export_dataset(self, dataset, filename):
        if self.format.lower() == "torch" or self.format.lower() == "pytorch":
            torch.save(dataset, filename)
            torch.save(dataset, filename)
        elif self.format.lower() == "tensorflow" or self.format.lower() == "tf":
            pass
        else:
            raise Exception
        pass

    def get_split_ratio(self, mode):
        return mode.split(":")

    def export_h5(self, filename, dataset_name, data_chunk, maxshape=None, chunks=True, rewrite=True,
                  compression="gzip"):
        """

        :param filename:
        :param dataset_name:
        :param data_chunk:
        :param maxshape:
        :param chunks:
        :param rewrite:
        :param compression:
        """
        # fcntl.flock(self.file_descriptor, fcntl.LOCK_EX)
        # print(data_chunk.shape)
        if rewrite:
            # Create an empty HDF5 file
            with h5py.File(filename, 'w') as f:
                if compression:
                    f.create_dataset(dataset_name, data=data_chunk, shape=data_chunk.shape, maxshape=maxshape,
                                     chunks=True,
                                     compression=compression)
                else:
                    f.create_dataset(dataset_name, data=data_chunk, shape=data_chunk.shape, maxshape=maxshape,
                                     chunks=True)
                pass
        else:
            with h5py.File(filename, 'a') as f:
                if chunks:
                    if dataset_name not in f:
                        if compression:
                            f.create_dataset(dataset_name, data=data_chunk, shape=data_chunk.shape, maxshape=maxshape,
                                             chunks=True, compression=compression)
                        else:
                            f.create_dataset(dataset_name, data=data_chunk, shape=data_chunk.shape, maxshape=maxshape,
                                             chunks=True)
                        # f[dataset_name]
                    else:
                        dataset = f[dataset_name]
                        current_shape = dataset.shape[0]
                        new_shape = current_shape + len(data_chunk)
                        dataset.resize(new_shape, axis=0)
                        dataset[current_shape:] = data_chunk
                else:
                    if dataset_name not in f:
                        if compression:
                            f.create_dataset(dataset_name, data=data_chunk, shape=data_chunk.shape, compression=compression)
                        else:
                            f.create_dataset(dataset_name, data=data_chunk, shape=data_chunk.shape, maxshape=maxshape,
                                             chunks=True)
        # self.mutex.release()
        # fcntl.flock(self.file_descriptor, fcntl.LOCK_UN)
        pass

    def run(self):
        # print(os.getpid())
        # read raw dataset from .h5 file
        # fprint("read raw dataset from .h5 file")
        # self.read_h5(self.key)

        with h5py.File(self.input_h5_file, 'r') as h5_file:
            # Assuming your dataset name inside the H5 file is 'data'
            # dataset_name = 'data'

            # Get the total number of elements in the dataset
            total_elements = h5_file[self.key].shape[0]

            # Initialize the start index and end index for chunk reading
            start_idx = 0

            # Read the dataset in chunks until the end
            while start_idx < total_elements:
                end_idx = min(start_idx + self.chunk_size, total_elements)

                # Read the chunk of data from the H5 file
                # chunk_data = h5_file[dataset_name][start_idx:end_idx]
                chunk_data = h5_file[self.key][start_idx:end_idx]
                chunk_label = h5_file[self.key+"_label"][start_idx:end_idx]
                # Process the chunk of data as needed
                # ...
                epi_datasets = EPIDatasets(chunk_data, chunk_label)

                # print(epi_datasets.data)
                ## split datasets
                fprint(msg="split datasets")
                if self.test_size is None:
                    if len(self.split_ratio) == 2:
                        self.test_size = self.split_ratio[1] / sum(self.split_ratio)

                train_X, test_X, train_y, test_y = train_test_split(epi_datasets.data, epi_datasets.label,
                                                                    test_size=self.test_size,
                                                                    random_state=self.random_state)
                # new_epi_datasets_train = (train_X, train_y)
                # new_epi_datasets_test = (test_X, test_y)
                # new_epi_datasets_train = EPIDatasets(train_X ,train_y)
                # new_epi_datasets_test = EPIDatasets(test_X, test_y)
                if start_idx == 0:
                    self.export_h5(self.output_prefix + "_train.h5", "X", train_X, rewrite=True,chunks=True,maxshape=(None,None,None),
                                   compression=self.compression)
                    self.export_h5(self.output_prefix + "_test.h5", "X", test_X, rewrite=True,chunks=True,maxshape=(None,None,None),
                                   compression=self.compression)


                else:
                    self.export_h5(self.output_prefix + "_train.h5", "X", train_X, rewrite=False, chunks=True,
                                   compression=self.compression)
                    self.export_h5(self.output_prefix + "_test.h5", "X", test_X, rewrite=False, chunks=True,
                                   compression=self.compression)

                self.export_h5(self.output_prefix + "_train.h5", "y", train_y, rewrite=False, chunks=True,maxshape=(None,None,None),
                               compression=self.compression)
                self.export_h5(self.output_prefix + "_test.h5", "y", test_y, rewrite=False, chunks=True,maxshape=(None,None,None),
                               compression=self.compression)
                # Move to the next chunk
                start_idx = end_idx
                fprint(msg="Memory usage: {}".format(self.memory_reporter.get_memory()))



        # ##
        # self.export_dataset(new_epi_datasets_train, self.output_prefix + "_train.pt")
        # self.export_dataset(new_epi_datasets_test, self.output_prefix + "_test.pt")
        pass
