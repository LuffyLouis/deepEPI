import concurrent
import os
import sys
sys.path.append("/mnt/d/zhouLab/CodeRepository/deepEPI")
from concurrent import futures

import numpy as np

import dna2vec.multi_k_model
from modules.utils import *

class DNA2VecEncoder:
    def __init__(self, pretrained_vec_file, k, threads, max_len=2000,pad_idx=0):
        self.pretrained_vec_model = dna2vec.multi_k_model.MultiKModel(pretrained_vec_file)
        self.k = k
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.threads = threads
        self.k_mers = []
        pass

    def fit(self, seq):
        self.k_mers = []
        ## generate the corresponding k-mers list
        if type(self.k) is list:
            for i in self.k:
                self.k_mers += self.generate_k_mers(seq, i)
        else:
            self.k_mers += self.generate_k_mers(seq, self.k)
        pass

    def transform(self, max_len):
        with futures.ThreadPoolExecutor(self.threads) as executor:  # 实例化线程池
            res = executor.map(self.get_vector_each, self.k_mers)
        res_array = np.vstack(list(res))

        return self.preprocess(res_array,max_len)

        # pass
    def preprocess(self,raw_array,max_len):
        if raw_array.shape[0] < max_len:
            pad_length = max_len - raw_array.shape[0]
            return np.r_[raw_array,np.full((pad_length,self.pretrained_vec_model.vec_dim),fill_value=self.pad_idx)]
        else:
            return raw_array[range(max_len)]
        pass

    def generate_k_mers(self, sequence, k):
        return generate_k_mers(sequence, k)
        # 设置线程池，可以根据需要调整线程数
        # with futures.ProcessPoolExecutor(max_workers=self.threads) as executor:
        #     # 将sequence切分成若干子序列，并在多个线程中并行生成k-mer序列
        #     chunk_size = len(sequence) // self.threads  # 假设将sequence平均切分成4个子序列
        #     chunks = [sequence[i:i + chunk_size] for i in range(0, len(sequence), chunk_size)]
        #     results = list(executor.map(generate_k_mers, chunks, [k] * len(chunks)))

        # # 合并结果
        # all_kmers = [kmer for result in results for kmer in result]
        # return all_kmers
        # pass

    def get_vector_each(self, k_mer):
        return self.pretrained_vec_model.vector(k_mer)
        pass


if __name__ == '__main__':
    timer = Timer()

    dna2vec_encoder = DNA2VecEncoder("../../input/dna2vec-20230705-0901-k3to8-100d-10c-32730Mbp-sliding-k0s.w2v",6,1)
    timer.start()
    dna2vec_encoder.fit("AGCTGATGCTGATGCTAGCTAGTTTAGCTTATTTCGTTAGCGTCTATCTATTCATCTTATTTTTCTAGTCATTCTATCGAGCTGATAGCTGATGCTGATGCTAGCTAGTTTAGCTTATTTCGTTAGCGTCTATCTATTCATCTTATTTTTCTAGTCATTCTATCGACGGCGATCGGAGCTGATGCTGATGCTAGCTAGTTTAGCTTATTTCGTTAGCGTCTATCTATTCATCTTATTTTTCTAGTCATTCTATCGACGGCGATCGGAGCTGATGCTGATGCTAGCTAGTTTAGCTTATTTCGTTAGCGTCTATCTATTCATCTTATTTTTCTAGTCATTCTATCGACGGCGATCGGGCTGATGCTAGCTAGTTTAGCTTATTTCGTTAGCGTCTATCTATTCATCTTATTTTTCTAGTCATTCTATCGACGGCGATCGGACGGCGATCGG")
    timer.stop()
    # res = dna2vec_encoder.transform()
    # print(res)
    # print(res.shape)

    print("Elapsed time: {}s".format(timer.elapsed_time()))
    # timer = Timer()

    # dna2vec_encoder = DNA2VecEncoder("../../input/dna2vec-20230705-0901-k3to8-100d-10c-32730Mbp-sliding-k0s.w2v", 6, 10)
    timer.start()
    dna2vec_encoder.fit("AGCTGATGCTGATGCTAGCTAGTTTAGCTTATTTCGTTAGCGTCAGCTGATGCTGATGCTAGCTAGTTTAGCTTATTTCGTTAGCGTCTATCTATTCATCTTATTTTTCTAGTCATTCTATCGACGGCGATCGGAGCTGATGCTGATGCTAGCTAGTTTAGCTTATTTCGTTAGCGTCTATCTATTCATCTTATTTTTCTAGTCATTCTATCGACGGCGATCGGAGCTGATGCTGATGCTAGCTAGTTTAGCTTATTTCGTTAGCGTCTATCTATTCATCTTATTTTTCTAGTCATTCTATCGACGGCGATCGGAGCTGATGCTGATGCTAGCTAGTTTAGCTTATTTCGTTAGCGTCTATCTATTCATCTTATTTTTCTAGTCATTCTATCGACGGCGATCGGTATCTATTCATCTTATTTTTCTAGTCATTCTATCGACGGCGATCGG")

    # res = dna2vec_encoder.transform()
    # print(res)
    # print(res.shape)
    timer.stop()
    print("Elapsed time: {}s".format(timer.elapsed_time()))

