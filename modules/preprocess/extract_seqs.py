import os
import re
import sys
import threading
import fcntl

import h5py
import numpy as np
import pyfastx
from sklearn.preprocessing import OneHotEncoder

from modules.preprocess.common import ChunkReadAndRunThread
from modules.utils import *

## Step to extract encoded datasets and corresponding fasta sequences
from modules.utils.encode import DNA2VecEncoder
from modules.utils.memory import MemoryUseReporter, convert_bytes_to_human_readable


class ExtractDatasets(ChunkReadAndRunThread):
    """
    Extract number encoded raw datasets (including inputs and labels) and corresponding fasta sequences

    """
    def __init__(self, balanced_interaction_file, raw_interaction_fasta_file, dataset_name, concat_reverse,
                 enhancer_length,promoter_length, trim, padding, compression, concat_epi, pretrained_vec_file, k_mer,
                 encode_method, temp_dir, output_dir, output, threads, chunk_size, verbose):
        super().__init__()

        # self.k_mer = k_mer
        self.timer = Timer()
        self.memory_use_reporter = MemoryUseReporter(os.getpid())

        if encode_method.lower() == "dna2vec":
            if pretrained_vec_file is None:
                fprint("ERROR",
                       "the pretrained vector file must be provided when the encode method was set to dna2vec!")
                # sys.exit(-1)
            self.pretrained_vec_file = pretrained_vec_file
            self.init_k_mer(k_mer)

        # 创建互斥锁
        # self.mutex = threading.Lock()
        # self.file_descriptor = os.open(output, os.O_RDWR | os.O_CREAT)

        self.balanced_interaction_file = balanced_interaction_file
        self.raw_interaction_fasta_file = raw_interaction_fasta_file
        self.concat_reverse = concat_reverse
        self.enhancer_length, self.trim, self.padding = int(enhancer_length), trim, padding
        self.promoter_length = int(promoter_length)
        self.encode_method = encode_method
        self.compression = compression
        self.concat_epi = concat_epi

        self.temp_dir = temp_dir
        check_and_create_dir(self.temp_dir)
        self.dataset_name = dataset_name
        self.output, self.threads, self.chunk_size = output, threads, chunk_size
        self.output_dir = output_dir
        if self.output_dir is None:
            self.output_dir = os.path.dirname(self.output)
        check_and_create_dir(self.output_dir)

        self.output_prefix = ".".join(os.path.basename(self.output).split(".")[0:-1])
        self.output_normal_fasta = os.path.join(self.output_dir, self.output_prefix + "_normal.fasta")
        self.output_reverse_fasta = os.path.join(self.output_dir, self.output_prefix + "_reverse.fasta")

        self.temp_output_dir = os.path.join(os.path.dirname(self.output), "temp_extract")
        check_and_create_dir(self.temp_output_dir)
        self.temp_output_pattern = "temp_extract_\d+"
        self.temp_output_prefix = "temp_extract_{}"

        self.verbose = verbose
        self.colnames = ["reads_name", "chrom1", "start1", "strand1", "chrom2", "start2", "strand2", "length",
                         "bin_name1", "bin_name2", "bin_index1", "bin_index2", "Unknown",
                         "compartment1", "compartment2", "compartment1_region", "compartment2_region", "end1", "end2",
                         "interaction_type", "promoter", "promoter_info", "distance", "within_compartment",
                         "filter", "label"]
        self.seq_encode_mode = np.array(list("NACTG")).reshape(-1, 1)
        if self.concat_epi:
            pass
        else:
            fprint("WARNING", "The EPI sequences would not be concat in output dataset!!")

    def init_interactions_fasta(self):
        """

        """
        self.raw_interactions_fasta = pyfastx.Fasta(self.raw_interaction_fasta_file, uppercase=True)
        pass

    def init_k_mer(self, kmer_str):
        """

        :param kmer_str:
        """
        if type(kmer_str) is str:
            self.k_mer = [int(i) for i in kmer_str.split(",")]
        else:
            self.k_mer = kmer_str
        # pass

    def extract_seq(self, fasta, unique_id, strand="+", reverse=False):
        """

        :param fasta:
        :param unique_id:
        :param strand:
        :param reverse:
        :return:
        """
        if reverse:
            if strand == "+":
                strand = "-"
            else:
                strand = "+"

        seq = fasta[unique_id]
        if strand == "+":
            # raw_seq = self.reference_genome_fasta.fetch(chrom,intervals,strand)
            return seq.seq
        else:
            return seq.antisense

    def init_encoder(self, method):
        """

        :param method:
        :return:
        """
        if method.lower() == "onehot" or method.lower() == "one-hot":
            encoder = OneHotEncoder()
            encoder.fit(self.seq_encode_mode)
            # encoder.tr
            return encoder
        elif method.lower() == "dna2vec":
            encoder = DNA2VecEncoder(self.pretrained_vec_file, self.k_mer, self.threads)
            return encoder
        pass

    def encode_seq(self, raw_seq, max_len=None):
        """

        :param raw_seq:
        :return:
        """
        if self.encode_method.lower() == "onehot":

            return self.encoder.transform(np.array(list(raw_seq)).reshape(-1, 1)).toarray()
        else:
            self.encoder.fit(raw_seq)
            return self.encoder.transform(max_len)

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
                                     chunks=True, dtype=data_chunk.dtype,
                                     compression=compression)
                else:
                    f.create_dataset(dataset_name, data=data_chunk, shape=data_chunk.shape, maxshape=maxshape,
                                     chunks=True, dtype=data_chunk.dtype)
                pass
        else:
            with h5py.File(filename, 'a') as f:
                if chunks:
                    if dataset_name not in f:
                        if compression:
                            f.create_dataset(dataset_name, data=data_chunk, shape=data_chunk.shape, maxshape=maxshape,
                                             chunks=True, dtype=data_chunk.dtype, compression=compression)
                        else:
                            f.create_dataset(dataset_name, data=data_chunk, shape=data_chunk.shape, maxshape=maxshape,
                                             chunks=True, dtype=data_chunk.dtype)
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
                            f.create_dataset(dataset_name, data=data_chunk, shape=data_chunk.shape,
                                             dtype=data_chunk.dtype, compression=compression)
                        else:
                            f.create_dataset(dataset_name, data=data_chunk, shape=data_chunk.shape, maxshape=maxshape,
                                             chunks=True, dtype=data_chunk.dtype)
        # self.mutex.release()
        # fcntl.flock(self.file_descriptor, fcntl.LOCK_UN)
        pass

    def process_seq(self, raw_seq, final_length):
        """

        :param raw_seq:
        :return:
        """
        if len(raw_seq) >= final_length:
            res_seq = trim_string_to_length(raw_seq, final_length, side=self.trim.lower())
        else:
            if self.padding.lower() == "left":
                res_seq = left_pad_string(raw_seq, final_length)
            elif self.padding.lower() == "right":
                res_seq = right_pad_string(raw_seq, final_length)
            else:
                res_seq = center_pad_string(raw_seq, final_length)
        return res_seq
        # pass

    def combine_interactions(self, chunk_id, output_file, temp_output_dir, temp_output_pattern, verbose):
        """

        :param chunk_id:
        :param output_file:
        :param temp_output_dir:
        :param temp_output_pattern:
        :param verbose:
        """
        all_files = os.listdir(temp_output_dir)
        all_valid_files = []
        count = 0
        for file in all_files:
            if len(re.findall(temp_output_pattern, file)) > 0:
                all_valid_files.append(file)
                count += 1
        fprint("LOG", "Finding {} temp output files".format(count))
        shape_total_raw = 0
        shape_total = 0
        for index, file_name in enumerate(all_valid_files):
            temp_file = os.path.join(temp_output_dir, file_name)
            with h5py.File(temp_file, "r") as h5_file:
                for i,dataset_name in enumerate(h5_file):
                    # print(dataset_name)
                    if verbose:
                        fprint("LOG", "Combing the {}-th file".format(index))
                    dataset = h5_file[dataset_name]
                    if dataset_name == "raw":
                        shape_total_raw += dataset.shape[0]
                        print("shape_total_raw:{}".format(shape_total_raw))
                    else:
                        shape_total += dataset.shape[0]
                        print("shape_total:{}".format(shape_total))
                    if index == 0 and i == 0:
                        self.export_h5(output_file, dataset_name, dataset, maxshape=(None, None, None), rewrite=True,
                                       compression=self.compression)
                    else:
                        self.export_h5(output_file, dataset_name, dataset, maxshape=(None, None, None), rewrite=False,
                                       compression=self.compression)
                    # dataset = h5_file[dataset_name]
                    # if dataset_name not in h5_file:
                    #     # 创建新的数据集并复制数据
                    #     h5_file.create_dataset(dataset_name, data=dataset[...])
                    # else:
                    #     # 已存在的数据集，将数据追加到已有数据集的末尾
                    #     h5_file[dataset_name].resize(h5_file[dataset_name].shape[0] + dataset.shape[0], axis=0)
                    #     h5_file[dataset_name][-dataset.shape[0]:] = dataset[...]

            os.remove(temp_file)
        pass

    def extract_dataset(self, chunk_id, task_id, temp_data, temp_file, dataset_name):
        """

        :param chunk_id:
        :param task_id:
        :param temp_data:
        :param temp_file:
        :param dataset_name:
        """
        # if chunk_id == 0 and task_id == 0:
        #     init = True
        # else:
        #     init = False

        # extract seqs
        data_chunk = []
        data_chunk_seq1 = []
        data_chunk_seq2 = []
        raw_interactions_fasta = pyfastx.Fasta(self.raw_interaction_fasta_file, uppercase=True)
        for i in range(temp_data.shape[0]):
            # rows = self.traverse_database(self.raw_interactions_db_manager, total_pages, self.chunksize, page_number)
            # for row in rows:
            # fasta_id = format_location(row)
            # print(temp_data)
            unique_id1 = format_location(
                [temp_data.loc[i, "chrom1"], temp_data.loc[i, "start1"], temp_data.loc[i, "end1"]])
            strand1 = temp_data.loc[i, "strand1"]
            unique_id2 = format_location(
                [temp_data.loc[i, "chrom2"], temp_data.loc[i, "start2"], temp_data.loc[i, "end2"]])
            strand2 = temp_data.loc[i, "strand2"]
            if temp_data.loc[i,"promoter"] == "0,1":
                enhancer_index = 0
                promoter_index = 1
            elif temp_data.loc[i, "promoter"] == "1,0":
                enhancer_index = 1
                promoter_index = 0
                pass
            else:
                enhancer_index = 0
                promoter_index = 1
                # fprint("WARNING",msg="it is not the enhancer-promoter pair for {}, so skip!!! ".format(temp_data.loc[i,"reads_name"]))
                # continue
                pass

            enhancer_id = [unique_id1, unique_id2][enhancer_index]
            promoter_id = [unique_id1, unique_id2][promoter_index]
            enhancer_strand = [strand1,strand2][enhancer_index]
            promoter_strand = [strand1,strand2][promoter_index]
            enhancer_seq = self.extract_seq(raw_interactions_fasta, enhancer_id, enhancer_strand)
            promoter_seq = self.extract_seq(raw_interactions_fasta, promoter_id, promoter_strand)

            ##
            seq1_encoded = self.encode_seq(self.process_seq(enhancer_seq, self.enhancer_length), self.enhancer_length)
            seq2_encoded = self.encode_seq(self.process_seq(promoter_seq, self.promoter_length), self.promoter_length)
            # print(seq1_encoded.toarray())
            # print("-----------------")
            # print(seq1_encoded.shape)
            # print(seq2_encoded.shape)
            # if self.encode_method.lower() == "onehot":
            if self.concat_epi:
                # fprint("WARNING","The EPI sequences would not be concat in output dataset!!")
                seq_concat = np.r_[seq1_encoded, seq2_encoded]
                # print(seq_concat.shape)
            # else:
            #     seq_concat = np.r_[seq1_encoded, seq2_encoded]

            if self.concat_reverse:
                print("asdsadsa")
                seq1_reverse = self.extract_seq(raw_interactions_fasta, enhancer_id, enhancer_strand, reverse=True)
                seq2_reverse = self.extract_seq(raw_interactions_fasta, promoter_id, promoter_strand, reverse=True)
                seq1_reverse_encoded = self.encode_seq(self.process_seq(seq1_reverse,self.enhancer_length), self.enhancer_length)
                seq2_reverse_encoded = self.encode_seq(self.process_seq(seq2_reverse,self.promoter_length), self.promoter_length)
                # if self.encode_method.lower() == "onehot":
                if self.concat_epi:
                    seq_reverse_concat = np.r_[seq1_reverse_encoded, seq2_reverse_encoded]
                # if self.concat_reverse:
                    seq_concat = np.c_[seq_concat, seq_reverse_concat]
                else:
                    seq1_encoded = np.c_[seq1_encoded, seq1_reverse_encoded]
                    seq2_encoded = np.c_[seq2_encoded, seq2_reverse_encoded]
                    # seq_concat = np.r_[seq_concat, seq_reverse_concat]

            if self.concat_epi:
                print(seq_concat.shape)
                data_chunk.append(seq_concat)
            else:
                data_chunk_seq1.append(seq1_encoded)
                data_chunk_seq2.append(seq2_encoded)
            # self.export_h5(output_file, dataset_name, seq_concat.shape, start_idx,init)
            # print(seq1)
            # break

        ## encode seqs and convert them into h5 format
        label = np.array(temp_data.loc[:, "label"]).reshape((-1, 1, 1))
        data_chunk = np.array(data_chunk)
        print(data_chunk.shape)
        data_chunk_seq1 = np.array(data_chunk_seq1)
        data_chunk_seq2 = np.array(data_chunk_seq2)

        # print(data_chunk)
        # self.mutex.acquire()
        if self.concat_epi:

            self.export_h5(temp_file, dataset_name, data_chunk, chunks=False, compression=None)

        else:
            self.export_h5(temp_file, dataset_name + "_enhancer", data_chunk_seq1, chunks=False, rewrite=False,
                           compression=None)
            self.export_h5(temp_file, dataset_name + "_promoter", data_chunk_seq2, chunks=False, rewrite=False,
                           compression=None)
        self.export_h5(temp_file, dataset_name + "_label", label, chunks=False, rewrite=False, compression=None)
        # self.mutex.release()
        pass

    def extract_seqs(self, index, temp_data):
        """

        :param index:
        :param temp_data:
        """
        if index == 0:
            self.output_func(self.output_normal_fasta, "", line_break=False, init=True)
            if self.concat_reverse:
                self.output_func(self.output_reverse_fasta, "", line_break=False, init=True)

        for i in range(temp_data.shape[0]):
            # rows = self.traverse_database(self.raw_interactions_db_manager, total_pages, self.chunksize, page_number)
            # for row in rows:
            # fasta_id = format_location(row)
            # print(temp_data)
            unique_id1 = format_location(
                [temp_data.loc[i, "chrom1"], temp_data.loc[i, "start1"], temp_data.loc[i, "end1"]])
            strand1 = temp_data.loc[i, "strand1"]
            unique_id2 = format_location(
                [temp_data.loc[i, "chrom2"], temp_data.loc[i, "start2"], temp_data.loc[i, "end2"]])
            strand2 = temp_data.loc[i, "strand2"]

            seq1 = self.extract_seq(self.raw_interactions_fasta, unique_id1, strand1)
            seq2 = self.extract_seq(self.raw_interactions_fasta, unique_id2, strand2)
            self.output_func(self.output_normal_fasta, self.format_fasta(unique_id1, seq1), line_break=False,
                             init=False)
            self.output_func(self.output_normal_fasta, self.format_fasta(unique_id2, seq2), line_break=False,
                             init=False)
            if self.concat_reverse:
                seq1_reverse = self.extract_seq(self.raw_interactions_fasta, unique_id1, strand1, reverse=True)
                seq2_reverse = self.extract_seq(self.raw_interactions_fasta, unique_id1, strand1, reverse=True)
                self.output_func(self.output_reverse_fasta, self.format_fasta(unique_id1, seq1_reverse),
                                 line_break=False,
                                 init=False)
                self.output_func(self.output_reverse_fasta, self.format_fasta(unique_id2, seq2_reverse),
                                 line_break=False,
                                 init=False)
            # print(seq1)
            # break
        pass

    def format_fasta(self, name, seq):
        """

        :param name:
        :param seq:
        :return:
        """
        return ">{}\n{}\n".format(name, seq)
        pass

    def run(self):
        self.timer.start()
        fprint(msg="Initializing the encoder {}...".format(self.encode_method))
        self.encoder = self.init_encoder(self.encode_method)
        self.init_interactions_fasta()
        ## extract sequences (plus strand/minus strand/both) from balanced interactions
        self.balanced_interactions_generator = self.read_valid_pairs(self.balanced_interaction_file, self.chunk_size,
                                                                     self.colnames, header=True)

        fprint(msg="Extracting normal and reverse fasta sequences from raw interaction fasta file...")
        for index, temp_data in enumerate(self.balanced_interactions_generator):
            self.extract_seqs(index, temp_data)

        # ##
        self.balanced_interactions_generator = self.read_valid_pairs(self.balanced_interaction_file, self.chunk_size,
                                                                     self.colnames, header=True)
        # print("sds")
        for chunk_id, temp_balanced_interaction_data in enumerate(self.balanced_interactions_generator):
            self.run_each(chunk_id, temp_balanced_interaction_data, self.threads, self.output, self.temp_output_dir,
                          self.temp_output_pattern,
                          fun_thread=self.extract_dataset, fun_args=(self.dataset_name,), verbose=self.verbose,
                          is_combine=True)
            pass

        self.timer.stop()
        fprint(msg="Complete done! Elapsed time: {:.3f}s".format(self.timer.elapsed_time()))
        # print(self.memory_use_list)
        fprint(msg="Memory usage at maximum: {}".format(convert_bytes_to_human_readable(max(self.memory_use_list))))
        pass
