# import sys
# import os
import re
import pybedtools
# import pandas as pd
from multiprocessing import Process
from modules.utils import *


class GenerateInteractions:
    def __init__(self, chrom_size_file, valid_pairs_file, temp_dir, eigenvec_dir, eigenvec_file_pattern, bin_size,
                 output_file,
                 threads, chunk_size, verbose,
                 keep_chrom_file=None, discard_chrom_file=None):
        self.thread_pool_list = []
        self.chrom_size_file, self.valid_pairs_file, self.eigenvec_dir, self.eigenvec_file_pattern, self.bin_size = \
            chrom_size_file, valid_pairs_file, eigenvec_dir, eigenvec_file_pattern, human_read_length(bin_size)
        self.keep_chrom_file, self.discard_chrom_file = keep_chrom_file, discard_chrom_file
        self.chunk_size = chunk_size
        self.temp_dir = temp_dir
        self.threads, self.verbose = threads, verbose
        self.output_file = output_file
        self.sqlite_table = 'combined_eigenvec'
        self.temp_output_dir, self.temp_output_pattern = os.path.join(os.path.dirname(output_file),
                                                                      "temp_output"), "combined_interactions_\d+.txt"
        # 创建一个计时器
        self.timer = Timer()

        check_and_create_dir(self.temp_dir)
        check_and_create_dir(self.temp_output_dir)
        # print(self.chrom_size_file)

    def annotate_eigenvec(self):
        ## read chrom size info
        chrom_size_data = read_chrom_size(self.chrom_size_file, return_type="raw")
        chrom_size_data = filt_chrom(chrom_size_data, self.keep_chrom_file, self.discard_chrom_file)
        # print(chrom_size_data)
        # chrom_size_data_dict = chrom_size_data.to_dict()
        eigenvec_data_list = []
        for file in os.listdir(self.eigenvec_dir):

            match_res = re.match(self.eigenvec_file_pattern, file)
            if match_res:

                chrom, bin_size = match_res.group(1), match_res.group(2)
                # print(chrom)
                # print(chrom_size_data.iloc[:,0].to_list())
                if chrom in chrom_size_data.iloc[:, 0].to_list():
                    chrom_size = int(chrom_size_data.loc[chrom_size_data.iloc[:, 0] == chrom, 1])
                    # print(bin_size == self.bin_size)
                    # print(bin_size)
                    # print(self.bin_size)

                    if self.bin_size and (bin_size == self.bin_size):
                        # bin_size = self.bin_size
                        bin_size = int(human_read_length(bin_size, reverse=True))
                        eigenvec_file = os.path.join(self.eigenvec_dir, file)
                        # print(eigenvec_file)
                        if os.path.getsize(eigenvec_file) > 0:
                            eigenvec_data = self.add_location(chrom, chrom_size, eigenvec_file, bin_size)
                            if type(eigenvec_data) == pd.DataFrame:
                                eigenvec_data_list.append(eigenvec_data)

        combined_eigen_data = self.combine_eigenvec(eigenvec_data_list)
        return combined_eigen_data
        # pass

    def read_valid_pairs(self, chunksize):
        """

        :param chunksize:
        :return:
        """
        valid_pairs_data_iterator = pd.read_table(self.valid_pairs_file,
                                                  names=["reads_name", "chrom1", "start1", "strand1", "chrom2",
                                                         "start2", "strand2", "length", "bin_name1", "bin_name2",
                                                         "bin_index1", "bin_index2", "Unknown"], chunksize=chunksize,
                                                  sep="\t")
        # valid_pairs_data.columns = ["reads_name","chrom1","start1","strand1","chrom2","start2","strand2","length","bin_name1","bin_name2","bin_index1","bin_index2","Unknown"]
        return valid_pairs_data_iterator

    @staticmethod
    def add_location(chrom, chrom_size, eigenvec_file, bin_size, sep="\t"):
        """

        :param chrom:
        :param chrom_size:
        :param eigenvec_file:
        :param bin_size:
        :param sep:
        :return:
        """
        # eigenvec_data: chrom, start, end, strand, eigen, compartment
        # print("eigenvec_file:{}".format(eigenvec_file))
        eigenvec_data = pd.read_table(eigenvec_file, header=None, sep=sep, engine='python')
        eigenvec_data.columns = ["eigen"]
        # eigenvec_data_copy = eigenvec_data.copy()
        # eigenvec_data
        starts = []
        ends = []
        # print(bin_size)
        # print(chrom_size)
        for start in range(0, chrom_size, bin_size):
            if abs(start - chrom_size) < bin_size:
                print("the last one bin")
                end = chrom_size

            else:
                end = start + bin_size
            starts.append(start)
            ends.append(end)
        # print(eigenvec_data.shape)
        # print(len(starts))
        eigenvec_data["chrom"] = chrom
        if len(starts) == eigenvec_data.shape[0]:
            eigenvec_data["start"] = starts
            eigenvec_data["end"] = ends
            eigenvec_data = eigenvec_data.reindex(columns=["chrom", "start", "end", "eigen"])
            eigenvec_data["compartment"] = None
            eigenvec_data.loc[eigenvec_data["eigen"] > 0, "compartment"] = "A"
            eigenvec_data.loc[eigenvec_data["eigen"] < 0, "compartment"] = "B"
            return eigenvec_data
        else:
            return None

    @staticmethod
    def combine_eigenvec(eigenvec_data_list):
        """

        :param eigenvec_data_list:
        :return:
        """
        combined_eigen_data = None
        for index in range(len(eigenvec_data_list)):
            if index == 0:
                combined_eigen_data = eigenvec_data_list[index]
            else:
                combined_eigen_data = pd.concat([combined_eigen_data, eigenvec_data_list[index]], axis=0)
        pass
        return combined_eigen_data

    @staticmethod
    def merge_interactions(df1, df2, annotation_col=None):
        # print(df2)
        if annotation_col is None:
            annotation_col = ["eigen", "compartment"]
        interaction1 = pybedtools.BedTool.from_dataframe(df1)
        interaction2 = pybedtools.BedTool.from_dataframe(df2)
        # 使用pybedtools进行合并操作
        merged_df = interaction2.intersect(interaction1, wb=True)
        # print("merged_df")
        # print(merged_df.head())
        # 将结果转换为DataFrame

        result_df = merged_df.to_dataframe(
            names=['chrom', 'start', 'end'] + annotation_col + ['chrom1', 'start1', 'end1', 'name', 'strand'])
        # print(result_df)
        result_df_filt = result_df.loc[:, ['chrom', 'start', 'end', 'name', 'strand'] + annotation_col]
        return result_df_filt
        pass

    def query_compartment(self, chrom, start, end, return_compartment=True):
        condition = "chrom = '{}' AND start <= '{}' AND end >= '{}'".format(chrom, start, end)
        res = self.db_manager.find_data(self.sqlite_table, condition)
        if return_compartment:
            if res.shape[0] == 1:
                return res["compartment"][0], format_location(res.loc[0, ["chrom", "start", "end"]].tolist())
            elif res.shape[0] > 1:
                return "Multi-compartment", ",".join(
                    [format_location(x) for x in res.loc[:, ["chrom", "start", "end"]].to_numpy()])
            else:
                return None, None
        # print("res:")
        # print(res)
        return res

        # pass

    @staticmethod
    def write_output(data, output, format="bed", init=False):
        """

        :param data:
        :param output:
        :param format:
        :param init:
        """
        if format == "csv":
            data.to_csv(output, index=False, header=init)
        else:
            data.to_csv(output, index=False, sep="\t", header=init)

    @staticmethod
    def output_func(output_file, content, line_break=True, init=True):
        """

        :param output_file:
        :param content:
        :param line_break:
        :param init:
        """
        if init:
            with open(output_file, 'w') as handle:
                pass
        else:

            if line_break:
                break_label = '\n'
            else:
                break_label = ''

            with open(output_file, 'a') as handle:
                handle.write("{}{}".format(content, break_label))
                pass

    # combine all temp gff file by each epoch
    def combine_interactions(self, chunk_id, output_file, temp_output_dir, temp_output_pattern, verbose):
        """

        :param chunk_id:
        :param output_file:
        :param temp_output_dir:
        :param temp_output_pattern:
        :param verbose:
        """
        all_files = os.listdir(temp_output_dir)
        count = 0
        for file in all_files:
            if len(re.findall(temp_output_pattern, file)) > 0:
                count += 1
        fprint("LOG", "Finding {} temp output files".format(count))
        if chunk_id == 0:
            self.output_func(output_file, '', init=True, line_break=False)
        for index in range(count):
            temp_file = os.path.join(temp_output_dir, temp_output_pattern.replace("\d+", str(index)))
            f = open(temp_file).read()
            if verbose:
                fprint("LOG", "Combing the {}-th file".format(index))
            self.output_func(output_file, f, init=False, line_break=False)
            os.remove(temp_file)
        pass

    def generate_interactions(self, chunk_id, temp_data, task_id, temp_file):
        """

        :param chunk_id:
        :param temp_data:
        :param task_id:
        :param temp_file:
        """
        if chunk_id == 0 and task_id == 0:
            init = True
        else:
            init = False
        temp_data["compartment1"] = None
        temp_data["compartment2"] = None
        temp_data["compartment1_region"] = None
        temp_data["compartment2_region"] = None

        temp_data["end1"] = temp_data["start1"] + temp_data["length"]
        temp_data["end2"] = temp_data["start2"] + temp_data["length"]

        for i in range(temp_data.shape[0]):
            chrom = temp_data.loc[i, "chrom1"]
            start = temp_data.loc[i, "start1"]
            end = temp_data.loc[i, "end1"]
            compartment1, compartment1_region = self.query_compartment(chrom, start, end)
            # print(res)
            temp_data.loc[i, "compartment1"] = compartment1
            temp_data.loc[i, "compartment1_region"] = compartment1_region

            chrom1 = temp_data.loc[i, "chrom2"]
            start1 = temp_data.loc[i, "start2"]
            end1 = temp_data.loc[i, "end2"]
            compartment2, compartment2_region = self.query_compartment(chrom1, start1, end1)
            temp_data.loc[i, "compartment2"] = compartment2
            temp_data.loc[i, "compartment2_region"] = compartment2_region
        # temp_data.to_csv(temp_file,index=False,header=True,sep="\t")
        # print("temp_data")
        # print(temp_data)
        self.write_output(temp_data, temp_file, format="txt", init=init)
        pass

    def run_each(self, chunk_id, valid_pairs_data, threads, output_file, temp_output_dir, temp_output_pattern, verbose):
        """

        :param chunk_id:
        :param valid_pairs_data:
        :param threads:
        :param output_file:
        :param temp_output_dir:
        :param temp_output_pattern:
        :param verbose:
        """
        # self.thread_pool_list = []

        # temp_data = valid_pairs_data.copy()
        # if type(valid_pairs_data) is pd.DataFrame:
        # print("valid_pairs_data.shape[0]")
        # print(valid_pairs_data.shape[0])
        records_tasks = initialize_all_tasks(valid_pairs_data.shape[0], threads)
        # thread_pool_list.append()
        # tasks = listen_batch(thread_pool_list)
        for index, task in enumerate(records_tasks):
            fprint(msg="Starting the {} epoch...".format(index))
            # execute the gene assign for each task
            task_id = index
            temp_data = valid_pairs_data.iloc[task, :]
            temp_data = temp_data.reset_index(drop=True)
            # print("temp_data")
            # print(temp_data)
            temp_file = os.path.join(temp_output_dir, temp_output_pattern.replace("\d+", str(index)))
            # print("temp_file:{}".format(temp_file))
            p = Process(target=self.generate_interactions, args=(chunk_id, temp_data, task_id, temp_file,))
            p.daemon = True
            p.start()
            self.thread_pool_list.append(p)
        while True:
            complete = listen_batch(self.thread_pool_list)
            # complete = True
            if complete:
                fprint(msg="Chunk {}: Complete done".format(chunk_id))
                self.combine_interactions(chunk_id, output_file, temp_output_dir, temp_output_pattern, verbose)
                break
        # if chunk_id == 0:
        #     temp_data.to_csv(output_file,index=False,header=True,sep="\t")
        # else:
        #     temp_data.to_csv(output_file, index=False, header=False, sep="\t", mode="a")
        # pd.DataFrame.to_csv("",)

    def run(self):
        self.timer.start()
        ## read eigenvector files and combine them
        combined_eigen_data = self.annotate_eigenvec()

        ## construct eigen sqlite3
        self.combined_eigen_db_path = os.path.join(self.temp_dir, "local.db")
        self.db_manager = DataFrameToSQLite(self.combined_eigen_db_path)
        self.db_manager.create_database()
        if not self.db_manager.is_table_exist(self.sqlite_table):
            self.db_manager.create_table(self.sqlite_table, combined_eigen_data)
        # self.db_manager.insert_data('combined_eigenvec', combined_eigen_data)
        # self.db_manager.find_data('combined_eigenvec',)
        # self.query_compartment()
        # self.db_manager.close_connection()

        ## read valid pairs data
        valid_pairs_data_iterator = self.read_valid_pairs(self.chunk_size)

        for chunk_id, chunk_data in enumerate(valid_pairs_data_iterator):
            fprint("Starting the chunk {}...".format(chunk_id))
            self.run_each(chunk_id, chunk_data, self.threads, self.output_file, self.temp_output_dir,
                          self.temp_output_pattern, self.verbose)

        self.db_manager.close_connection()
        self.timer.stop()
        fprint("Complete done! Elapsed time: {:.3f}s".format(self.timer.elapsed_time()))


if __name__ == '__main__':
    genereateInteractions = GenerateInteractions(chrom_size_file="../../sample/hg38.chrom_size",
                                                 valid_pairs_file="../../sample/SRR400264_00.allValidPairs",
                                                 eigenvec_dir="../../", eigenvec_file_pattern="eigen_(.*)_(.*).txt",
                                                 bin_size=500000, chunk_size=1e5, output_file="./annotated_interactions.txt",
                                                 threads=20, temp_dir="./temp",
                                                 discard_chrom_file="../../sample/keep_chrom.txt", verbose=True)
    genereateInteractions.run()
