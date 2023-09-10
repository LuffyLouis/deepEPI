
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
from modules import trim_string_to_length, left_pad_string, right_pad_string, center_pad_string, Timer
from modules.models.models import SimpleCNN, EPIMind
from modules.models.training_model import load_weigths
from modules.utils.encode import DNA2VecEncoder
from modules.utils.tools import check_and_create_dir, fprint


class PredictEPI:
    def __init__(self,model,
                 encode_method,concat_reverse,enhancer_len,promoter_len,num_heads,num_layers,num_hiddens,ffn_num_hiddens,
                 concat_epi,trim, padding,k_mer,pretrained_vec_file,threads,
                 save_param_dir, save_param_prefix,enhancer_seq,promoter_seq,ref_file=None,alt_file=None,device="cuda",verbose=True):
        ##
        self.timer = Timer()
        self.timer.final_start()
        ##
        self.model = model
        self.save_param_dir = save_param_dir
        self.encode_method = encode_method
        self.concat_reverse = concat_reverse

        ##

        # self.enhancer_length = enhancer_len
        self.concat_epi = concat_epi
        self.pretrained_vec_file = pretrained_vec_file
        self.threads = threads
        self.k_mer = k_mer

        ## seq process
        self.trim, self.padding = trim, padding
        ## EPI-Mind
        self.enhancer_length = int(enhancer_len)
        self.promoter_length = int(promoter_len)
        self.num_heads,self.num_layers,self.num_hiddens,self.ffn_num_hiddens = num_heads,num_layers,num_hiddens,ffn_num_hiddens

        self.save_param_prefix, self.enhancer_seq, self.promoter_seq, self.ref_file, self.alt_file, self.device = \
            save_param_prefix,enhancer_seq,promoter_seq,ref_file,alt_file,device

        self.verbose = verbose

        self.predict_model = None
        self.seq_encode_mode = np.array(list("NACTG")).reshape(-1, 1)

        ##
        self.encoder = self.init_encoder(self.encode_method)
        ## loading model
        self.__load_model()
        self.__load_weights()
        pass

    def __load_model(self):
        if self.model.lower() == "simplecnn":
            self.predict_model = SimpleCNN(encode_method=self.encode_method, concat_reverse=self.concat_reverse,verbose=self.verbose)
        elif self.model.lower() == "epimind":
            self.predict_model = EPIMind(concat_reverse=self.concat_reverse,
                                          enhancer_len=self.enhancer_length, promoter_len=self.promoter_length,
                                          num_heads=self.num_heads, num_layers=self.num_layers,
                                          num_hiddens=self.num_hiddens, ffn_num_hiddens=self.ffn_num_hiddens,verbose=self.verbose)
            pass
        else:
            sys.exit(-1)
        pass

    def __load_weights(self):
        if self.save_param_dir is not None and self.save_param_prefix is not None:
            check_and_create_dir(self.save_param_dir)
            loading_res = load_weigths(self.save_param_dir, self.save_param_prefix)
            # print("loading_res: {}".format(loading_res))
            if loading_res:
                fprint(msg="Loading the latest weights from {} directory based on {} prefix...".format(self.save_param_dir, self.save_param_prefix))
                self.predict_model.load_state_dict(loading_res, strict=False)
            else:
                pass
        else:
            fprint(msg="No specify the valid weights file, so the model will predict with initialized weights!!!")
            pass
        pass
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

    def extract_seq(self,seq):
        from Bio.Seq import Seq
        # from Bio.Alphabet import IUPAC
        seq_obj = Seq(seq)
        return seq_obj.reverse_complement()
        pass

    def predict(self,enhancer_seq,promoter_seq):
        data_chunk = []
        data_chunk_seq1,data_chunk_seq2 = [],[]
        seq1_encoded = self.encode_seq(self.process_seq(enhancer_seq, self.enhancer_length), self.enhancer_length)
        seq2_encoded = self.encode_seq(self.process_seq(promoter_seq, self.promoter_length), self.promoter_length)

        if self.concat_epi:
            # fprint("WARNING","The EPI sequences would not be concat in output dataset!!")
            seq_concat = np.r_[seq1_encoded, seq2_encoded]

        if self.concat_reverse:
            seq1_reverse = self.extract_seq(enhancer_seq)
            seq2_reverse = self.extract_seq(promoter_seq)
            seq1_reverse_encoded = self.encode_seq(self.process_seq(seq1_reverse, self.enhancer_length),
                                                   self.enhancer_length)
            seq2_reverse_encoded = self.encode_seq(self.process_seq(seq2_reverse, self.promoter_length),
                                                   self.promoter_length)
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
            # print(seq_concat.shape)
            data_chunk.append(seq_concat)
        else:
            data_chunk_seq1.append(seq1_encoded)
            data_chunk_seq2.append(seq2_encoded)
        data_chunk = torch.tensor(np.array(data_chunk),dtype=torch.float32)
        # print(data_chunk.shape)
        # data_chunk_seq1 = np.array(data_chunk_seq1)
        # data_chunk_seq2 = np.array(data_chunk_seq2)
        fprint(msg="Predicting......")
        with torch.no_grad():
            self.predict_model.eval()
            pred_res = self.predict_model(data_chunk)

        return pred_res

    def batch_predict(self,enhancer_seq_list,promoter_seq_list):
        # enhancer_seq_list = self.enhancer_seq.split(",")
        # promoter_seq_list = self.promoter_seq.split(",")
        self.timer.start()
        fprint(msg="Encoding......")
        data_chunk = []
        for enhancer_seq,promoter_seq in zip(enhancer_seq_list,promoter_seq_list):

            data_chunk_seq1, data_chunk_seq2 = [], []
            seq1_encoded = self.encode_seq(self.process_seq(enhancer_seq, self.enhancer_length), self.enhancer_length)
            seq2_encoded = self.encode_seq(self.process_seq(promoter_seq, self.promoter_length), self.promoter_length)

            if self.concat_epi:
                # fprint("WARNING","The EPI sequences would not be concat in output dataset!!")
                seq_concat = np.r_[seq1_encoded, seq2_encoded]

            if self.concat_reverse:
                seq1_reverse = self.extract_seq(enhancer_seq)
                seq2_reverse = self.extract_seq(promoter_seq)
                seq1_reverse_encoded = self.encode_seq(self.process_seq(seq1_reverse, self.enhancer_length),
                                                       self.enhancer_length)
                seq2_reverse_encoded = self.encode_seq(self.process_seq(seq2_reverse, self.promoter_length),
                                                       self.promoter_length)
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
                # print(seq_concat.shape)
                data_chunk.append(seq_concat)
            else:
                data_chunk_seq1.append(seq1_encoded)
                data_chunk_seq2.append(seq2_encoded)
        data_chunk = torch.tensor(np.array(data_chunk), dtype=torch.float32)
        # print(data_chunk.shape)
        # data_chunk_seq1 = np.array(data_chunk_seq1)
        # data_chunk_seq2 = np.array(data_chunk_seq2)
        fprint(msg="Predicting......")
        with torch.no_grad():
            self.predict_model.eval()
            pred_res = self.predict_model(data_chunk)

        self.timer.stop()
        fprint(msg="Time cost: {:.3f}s/batch".format(self.timer.elapsed_time()))
        return pred_res
        # suffix = os.path.basename(self.enhancer_file).split(".")[-1]
        # if suffix.lower() == "csv":
        #     enhancer_data = pd.read_table(self.enhancer_file,sep=",")
        #     promoter_data = pd.read_table(self.promoter_file, sep=",")
        #     for

        #

        pass

    def run(self):
        # import csv
        if self.ref_file or self.alt_file:
            if self.ref_file:
                suffix = os.path.basename(self.ref_file).split(".")[-1]
                if suffix.lower() == "csv":

                    ref_data = pd.read_table(self.ref_file,sep=",")
                else:
                    ref_data = pd.read_table(self.ref_file, sep="\t")
                print(ref_data)
                enhancer_seq_list = ref_data.iloc[:, 0].to_list()
                promoter_seq_list = ref_data.iloc[:, 1].to_list()
                ref_predict = self.batch_predict(enhancer_seq_list,promoter_seq_list)
                print(ref_predict)
            if self.alt_file:
                suffix = os.path.basename(self.alt_file).split(".")[-1]
                if suffix.lower() == "csv":

                    alt_data = pd.read_table(self.alt_file, sep=",")
                else:
                    alt_data = pd.read_table(self.alt_file, sep="\t")
                # alt_data = pd.read_table(self.alt_file, sep=",")
                enhancer_seq_list = alt_data.iloc[:, 0].to_list()
                promoter_seq_list = alt_data.iloc[:, 1].to_list()
                alt_predict = self.batch_predict(enhancer_seq_list, promoter_seq_list)
                print(alt_predict)
                # self.batch_predict(enhancer_seq_list,promoter_seq_list)
                pass

            pass
        else:
            self.timer.start()
            if self.enhancer_seq and self.promoter_seq:
                # if "," in self.enhancer_seq and "," in self.promoter_seq:
                enhancer_seq_list = self.enhancer_seq.split(",")
                promoter_seq_list = self.promoter_seq.split(",")

                for enhancer, promoter in zip(enhancer_seq_list,promoter_seq_list):
                    pred_res = self.predict(enhancer,promoter)
                    fprint(msg="Predict result: {}".format(pred_res[:,1]))
                # print(pred_res[:,1])
                pass
            self.timer.stop()
            fprint(msg="Time cost: {:.3f}s".format(self.timer.elapsed_time()))
        self.timer.final_stop()
        fprint(msg="Time cost in total: {:.3f}s".format(self.timer.final_elapsed_time()))
        pass