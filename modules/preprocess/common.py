import os
import re
import sys
import time
from multiprocessing import Process

import pandas as pd

from modules import fprint, initialize_all_tasks, listen_batch
from modules.utils.memory import MemoryUseReporter


class ChunkReadAndRunThread:
    """
    A class for reading data by a chunk size with multi-threads/processors
    """
    def __init__(self,report_interval=1):
        # self.valid_pairs_file = valid_pairs_file
        self.memory_use_reporter = MemoryUseReporter(os.getpid())
        self.memory_use_list = []
        self.report_interval = report_interval
        pass

    @staticmethod
    def read_valid_pairs(valid_pairs_file, chunksize, colnames, header=None):
        """

        :param valid_pairs_file:
        :param chunksize:
        :param colnames:
        :param header:
        :return:
        """
        if header:
            valid_pairs_data_iterator = pd.read_table(valid_pairs_file,
                                                      names=colnames,
                                                      skiprows=1,
                                                      chunksize=chunksize, sep="\t")
        else:
            valid_pairs_data_iterator = pd.read_table(valid_pairs_file,
                                                      names=colnames,
                                                      # skiprows=1,
                                                      chunksize=chunksize, sep="\t")
        # valid_pairs_data.columns = ["reads_name","chrom1","start1","strand1","chrom2","start2","strand2","length","bin_name1","bin_name2","bin_index1","bin_index2","Unknown"]
        return valid_pairs_data_iterator

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

    def combine_interactions(self, chunk_id, output_file, temp_output_dir, temp_output_pattern, verbose):
        """

        :param chunk_id: chunk id for reading data with a chunk size
        :param output_file: the path of output file
        :param temp_output_dir: temporary directory
        :param temp_output_pattern: the RegExp pattern for temporary filename
        :param verbose: whether to open the verbose mode
        """
        all_files = os.listdir(temp_output_dir)
        all_valid_files = []
        count = 0
        for file in all_files:
            if len(re.findall(temp_output_pattern, file)) > 0:
                all_valid_files.append(file)
                count += 1
        fprint("LOG", "Finding {} temp output files".format(count))
        all_valid_files.sort()
        if chunk_id == 0:
            self.output_func(output_file, '', init=True, line_break=False)
        for index, file in enumerate(all_valid_files):
            temp_file = os.path.join(temp_output_dir, file)
            f = open(temp_file).read()
            if verbose:
                fprint("LOG", "Combing the {}-th file".format(index))
            self.output_func(output_file, f, init=False, line_break=False)
            os.remove(temp_file)

    @staticmethod
    def write_output(data, output, format="bed", init=False):
        if format == "csv":
            data.to_csv(output, index=False, header=init)
        else:
            data.to_csv(output, index=False, sep="\t", header=init)

    def run_each(self, chunk_id, valid_pairs_data, threads, output_file, temp_output_dir, temp_output_pattern,
                 fun_thread, fun_args, verbose, is_combine=True):
        """

        :param chunk_id:
        :param valid_pairs_data:
        :param threads:
        :param output_file:
        :param temp_output_dir:
        :param temp_output_pattern:
        :param fun_thread:
        :param fun_args:
        :param verbose:
        """
        self.thread_pool_list = []
        records_tasks = initialize_all_tasks(valid_pairs_data.shape[0], threads)
        # thread_pool_list.append()
        # tasks = listen_batch(thread_pool_list)
        for index, task in enumerate(records_tasks):
            fprint(msg="Starting the {} epoch...".format(index))
            # execute the gene assign for each task
            task_id = index
            temp_data = valid_pairs_data.iloc[task, :]
            temp_data = temp_data.reset_index(drop=True)
            # print(temp_data)
            temp_file = os.path.join(temp_output_dir, temp_output_pattern.replace("\d+", str(index)))
            # print("temp_file:{}".format(temp_file))
            if is_combine:
                p = Process(target=fun_thread, args=(chunk_id, task_id, temp_data, temp_file) + fun_args)
            else:
                p = Process(target=fun_thread, args=(chunk_id, task_id, temp_data) + fun_args)
            p.daemon = True
            p.start()
            self.thread_pool_list.append(p)
        while True:
            complete = listen_batch(self.thread_pool_list)

            fprint(msg="Memory use: {}".format(self.memory_use_reporter.get_memory()))
            self.memory_use_list.append(self.memory_use_reporter.get_memory())
            time.sleep(self.report_interval)
            # complete = True
            if complete:
                fprint(msg="Chunk {}: Complete done".format(chunk_id))
                if is_combine:
                    self.combine_interactions(chunk_id, output_file, temp_output_dir, temp_output_pattern, verbose)
                break
            # elif complete is None:
            #     sys.exit(-1)

