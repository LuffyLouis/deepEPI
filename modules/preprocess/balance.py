import math
import os
import sqlite3
import sys
from multiprocessing import Process

import Bio
import pandas as pd
import pyfastx

# from modules import DataFrameToSQLite, format_location, fprint, mean
from modules.utils import *
from modules.preprocess.common import ChunkReadAndRunThread


class BalanceDatasets(ChunkReadAndRunThread):
    """
    Balance the sequence datasets at positive and negative section

    """
    def __init__(self, filter_interactions_file, raw_interactions_file,reference_genome, GC_content,
                 balanced_interaction_type,balanced_same_chromsome,
                 threads,  chunksize, temp_dir, output,verbose):
        super().__init__()

        self.reference_genome_fasta = None
        self.filter_interactions_file, self.raw_interactions_file, self.GC_content, self.verbose = \
            filter_interactions_file, raw_interactions_file, GC_content, verbose
        self.balanced_interaction_type = balanced_interaction_type
        self.balanced_same_chromsome = balanced_same_chromsome
        self.reference_genome = reference_genome
        self.chunksize = chunksize
        self.output_file = output
        self.output_dir, self.output_prefix = os.path.dirname(output),".".join(os.path.basename(output).split(".")[0:-1])
        self.temp_dir = temp_dir
        self.threads = threads
        self.temp_output_dir, self.temp_output_prefix = os.path.join(self.output_dir,"temp_balanced"),self.output_prefix + "_{}"
        check_and_create_dir(self.temp_output_dir)
        self.temp_output_pattern = self.output_prefix + "\d+"
        self.colnames = ["reads_name", "chrom1", "start1", "strand1", "chrom2", "start2", "strand2", "length",
                         "bin_name1", "bin_name2", "bin_index1", "bin_index2", "Unknown",
                         "compartment1", "compartment2", "compartment1_region", "compartment2_region", "end1", "end2",
                         "interaction_type",    "promoter",    "promoter_info",    "distance",    "within_compartment",
                         "filter"]
        self.raw_interactions_db_path = os.path.join(self.temp_dir,"local.db")
        self.raw_interactions_table = "raw_interactions"
        self.raw_interactions_fasta = "raw_interactions.fasta"
        self.negative_interactions = set()

        self.raw_total = 0
        self.pos = 0
        self.neg = 0
        ##
        self.timer = Timer()
        # self.positive_file = self.output_prefix + "_pos.txt"
        # self.negative_file = self.output_prefix + "_neg.txt"

    def get_GC_content(self, seq):
        """

        :param seq:
        :return:
        """
        gc_count = seq.count("G") + seq.count("C")
        return gc_count / len(seq)

    def get_mean_GC_content(self):
        pass

    def extract_fasta(self,chunk_id, temp_data, return_GC=True):
        """

        :param chunk_id:
        :param temp_data:
        :param return_GC:
        :return:
        """
        self.temp_interactions_fasta_file = os.path.join(self.temp_output_dir,self.raw_interactions_fasta)
        # total_pages = self.raw_interactions_db_manager.get_total_pages(self.raw_total,self.chunksize)
        if chunk_id == 0:
            self.output_func(self.temp_interactions_fasta_file, "", line_break=False,init=True)
        # temp_data["GC_content"] = None
        # temp_data["GC_content"].apply()
        mean_GC_contents = []
        for i in range(temp_data.shape[0]):
            # rows = self.traverse_database(self.raw_interactions_db_manager, total_pages, self.chunksize, page_number)
            # for row in rows:
            # fasta_id = format_location(row)
            # print(temp_data)
            seq1 = self.extract_seq(self.reference_genome_fasta, temp_data.loc[i,"chrom1"], temp_data.loc[i,"start1"], temp_data.loc[i,"end1"])
            seq2 = self.extract_seq(self.reference_genome_fasta, temp_data.loc[i,"chrom2"], temp_data.loc[i,"start2"], temp_data.loc[i,"end2"])
            # print(seq1)
            # break
            seq1_fasta = self.format_fasta(seq1.name,seq1.seq)
            seq2_fasta = self.format_fasta(seq2.name, seq2.seq)
            self.output_func(self.temp_interactions_fasta_file, seq1_fasta, line_break=False,init=False)
            self.output_func(self.temp_interactions_fasta_file, seq2_fasta, line_break=False, init=False)
            # print(type(seq1.gc_content))
            mean_GC_content = mean([seq1.gc_content, seq2.gc_content])
            mean_GC_contents.append(mean_GC_content)
        if return_GC:
            return mean_GC_contents
        else:
            return None

    def create_raw_interaction_db(self):
        """

        """
        self.raw_interaction_generator = self.read_valid_pairs(self.raw_interactions_file,self.chunksize,self.colnames,header=True)
        self.raw_interactions_db_manager = DataFrameToSQLite(self.raw_interactions_db_path)
        self.raw_interactions_db_manager.create_database()
        for index, temp_data in enumerate(self.raw_interaction_generator):
            if not self.raw_interactions_db_manager.is_table_exist(self.raw_interactions_table):
                ## Extract the raw sequence of all raw interactions
                if self.verbose:
                    fprint(msg="Extract raw fasta of all interactions...")
                mean_GC_contents = self.extract_fasta(index,temp_data)

                ## Calculate the GC content
                temp_data["GC_content"] = mean_GC_contents
                # temp_data["GC_content"] = temp_data["GC_content"].apply()
                if index == 0:
                    self.raw_interactions_db_manager.create_table(self.raw_interactions_table, temp_data)
                else:
                    self.raw_interactions_db_manager.insert_data(self.raw_interactions_table, temp_data)

                self.raw_total += temp_data.shape[0]
            else:
                self.raw_total = self.raw_interactions_db_manager.count_entries(self.raw_interactions_table)
                break

    # def create_filter_interaction_db(self):
    #     self.filter_interaction_generator = self.read_valid_pairs(self.filter_interactions_file, self.chunksize,
    #                                                            self.colnames, header=True)
    #     self.raw_interactions_db_manager = DataFrameToSQLite(self.raw_interactions_db_path)
    #     self.raw_interactions_db_manager.create_database()
    #     for index, temp_data in enumerate(self.raw_interaction_generator):
    #         if not self.raw_interactions_db_manager.is_table_exist(self.raw_interactions_table):
    #             if index == 0:
    #                 self.raw_interactions_db_manager.create_table(self.raw_interactions_table, temp_data)
    #             else:
    #                 self.raw_interactions_db_manager.insert_data(self.raw_interactions_table, temp_data)
    #             self.raw_total += temp_data.shape[0]
    #         else:
    #             self.raw_total = self.raw_interactions_db_manager.count_entries(self.raw_interactions_table)
    #             break
        # return True
    def init_genome(self):
        """

        """
        self.reference_genome_fasta = pyfastx.Fasta(self.reference_genome, uppercase=True)
        pass

    def create_interactions_db(self):

        pass

    def init_interactions(self):
        """

        """
        self.interactions_fasta = pyfastx.Fasta(self.temp_interactions_fasta_file, uppercase=True)
        pass

    def extract_seq(self,fasta,chrom,start,end,strand="+"):
        """

        :param fasta:
        :param chrom:
        :param start:
        :param end:
        :param strand:
        :return:
        """
        seq = fasta[chrom][start - 1:end]
        if strand == "+":
            # raw_seq = self.reference_genome_fasta.fetch(chrom,intervals,strand)
            return seq
        else:
            return seq.antisense

    def query_matched_neg(self, chrom1,start1,end1,
                                    chrom2,start2,end2):
        """

        :param chrom1:
        :param start1:
        :param end1:
        :param chrom2:
        :param start2:
        :param end2:
        :return:
        """
        condition = "chrom1 = '{}' AND start1 == '{}' AND end1 == '{}' AND chrom2 = '{}' AND start2 == '{}' AND end2 == '{}'".\
            format(chrom1, start1, end1, chrom2, start2, end2)
        res = self.raw_interactions_db_manager.find_data(self.raw_interactions_table,condition)
        if res.shape[0] >= 1:
            GC_content_pos = res.loc[0,"GC_content"]
            compartment1_pos = res.loc[0, "compartment1"]
            compartment2_pos = res.loc[0, "compartment2"]
            # if self.balanced_same_chromsome and self.balanced_interaction_type == "intra":
            #     chrom_pos = res.loc[0,"chrom1"]
            if self.balanced_interaction_type.lower() != "all":
                if self.balanced_interaction_type == "intra" and self.balanced_same_chromsome:
                    condition2 = "chrom1 = '{}' AND filter = '0' AND interaction_type = '{}' AND compartment1 = '{}' AND compartment2 = '{}' AND ABS(GC_content - '{}') <= {} ORDER BY ABS(GC_content - '{}') LIMIT 1000". \
                        format(chrom1, self.balanced_interaction_type.lower(), compartment1_pos, compartment2_pos,
                               GC_content_pos, self.GC_content, GC_content_pos)
                else:
                    condition2 = "filter = '0' AND interaction_type = '{}' AND compartment1 = '{}' AND compartment2 = '{}' AND ABS(GC_content - '{}') <= {} ORDER BY ABS(GC_content - '{}') LIMIT 1000".\
                        format(self.balanced_interaction_type.lower(),compartment1_pos,compartment2_pos,GC_content_pos,self.GC_content,GC_content_pos)
            else:
                condition2 = "filter = '0' AND compartment1 = '{}' AND compartment2 = '{}' AND ABS(GC_content - '{}') <= {} ORDER BY ABS(GC_content - '{}') LIMIT 1000". \
                    format(compartment1_pos, compartment2_pos, GC_content_pos,
                           self.GC_content, GC_content_pos)
            matched_results = self.raw_interactions_db_manager.find_data(self.raw_interactions_table, condition2)
            # print(matched_results)
            matched_results["filter"] = False
            if matched_results.shape[0] >= 1:
                # print(matched_results)
                for i in range(matched_results.shape[0]):
                    negative_interaction = "|".join([matched_results.loc[i,"bin_name1"],matched_results.loc[i,"bin_name2"]])
                    if negative_interaction not in self.negative_interactions:
                        self.negative_interactions.add(negative_interaction)
                    return matched_results.loc[i,:]

                return None
            else:
                fprint("Warning","No found the most matched negative interaction in raw interactions!!!")
                return None
        return None

        pass

    def balance_datasets(self,chunk_id, task_id, temp_data, temp_file
                         ):
        """

        :param chunk_id:
        :param task_id:
        :param temp_data:
        :param temp_file:
        """
        if chunk_id == 0 and task_id == 0:
            init = True
        else:
            init = False

        temp_data_pos = temp_data.copy()
        # temp_data_pos["label"] = 1
        temp_data_negative = pd.DataFrame(columns=temp_data_pos.columns)
        # temp_data_pos["label"] = 0
        ##
        for i in range(temp_data_pos.shape[0]):
            chrom1 = temp_data_pos.loc[i,"chrom1"]
            start1 = temp_data_pos.loc[i,"start1"]
            end1 = temp_data_pos.loc[i,"end1"]
            chrom2 = temp_data_pos.loc[i,"chrom2"]
            start2 = temp_data_pos.loc[i,"start2"]
            end2 = temp_data_pos.loc[i,"end2"]

            # pos_gc = mean([self.extract_seq(self.raw_interactions_fasta,chrom1,start1,end1).gc_content,
            #                     self.extract_seq(self.raw_interactions_fasta,chrom2,start2,end2).gc_content])
            # pos_gc = temp_data_pos[""]
            # print(chrom1)
            matched_negative_interaction = self.query_matched_neg(chrom1,start1,end1,
                                    chrom2,start2,end2)
            temp_data_negative.loc[i,:] = matched_negative_interaction
        temp_data_pos["label"] = 1
        temp_data_negative["label"] = 0
        temp_data_filt = pd.concat([temp_data_pos,temp_data_negative],axis=0)
        self.write_output(temp_data_filt, temp_file, format="txt", init=init)
        pass

    def run(self):
        self.timer.start()
        ## initialize
        fprint(msg="Initialize the reference genome...")
        self.init_genome()

        ## create the sqlite database for raw interactions
        fprint(msg="Create the reference genome sqlite database...")
        self.create_raw_interaction_db()

        ## extract seqs
        # fprint("Extract raw fasta of all interactions...")
        # self.extract_fasta()

        ## find the less dataset as query to search the most matched record with similar GC content in the sqlite database

        ##
        # self.init_interactions()
        self.filter_interactions_generator = self.read_valid_pairs(self.filter_interactions_file,self.chunksize,self.colnames,header=True)
        for chunk_id, chunk_data in enumerate(self.filter_interactions_generator):
            self.run_each(chunk_id,chunk_data,self.threads,self.output_file,self.temp_output_dir,self.temp_output_pattern,
                          fun_thread=self.balance_datasets,fun_args=(),verbose=self.verbose)
            # self.thread_pool_list.append(p)
        self.timer.stop()
        fprint(msg="Complete done! Elapsed time: {:.3f}s".format(self.timer.elapsed_time()))
        # pass

    # def extract_fasta_from_db(self):
    #     self.temp_interactions_fasta_file = os.path.join(self.temp_output_dir,self.raw_interactions_fasta)
    #     total_pages = self.raw_interactions_db_manager.get_total_pages(self.raw_total,self.chunksize)
    #     self.output_func(self.temp_interactions_fasta_file, "", line_break=False,init=True)
    #     for page_number in range(1, total_pages+1):
    #         rows = self.traverse_database(self.raw_interactions_db_manager, total_pages, self.chunksize, page_number)
    #         for row in rows:
    #             # fasta_id = format_location(row)
    #             seq = self.extract_seq(self.reference_genome_fasta,row[0],row[1],row[2])
    #
    #             self.output_func(self.temp_interactions_fasta_file,seq.raw, line_break=False,init=False)
    #     pass

    # def traverse_database(self, db_manager, total_pages, page_size, page_number):
    #     # total_records = get_total_records(db_file)
    #     # total_pages = get_total_pages(total_records, page_size)
    #
    #     # 根据总页数和特定页数进行页数判断和处理
    #     if page_number < 1 or page_number > total_pages:
    #         print("Invalid page number. Please choose a valid page.")
    #         return
    #
    #     # conn = sqlite3.connect(db_file)
    #     cursor = db_manager.conn.cursor()
    #
    #     # try:
    #     offset = (page_number - 1) * page_size
    #     cursor.execute(f"SELECT chrom, start, end FROM your_table_name LIMIT {page_size} OFFSET {offset};")
    #     rows = cursor.fetchall()
    #     return rows
            # for row in rows:
            #     chrom, start, end = row
            #     print(f"Chromosome: {chrom}, Start: {start}, End: {end}")

        # except sqlite3.Error as e:
        #     print(f"Error occurred: {e}")
        #
        # finally:
        #     conn.close()
    def format_fasta(self, name, seq):
        return ">{}\n{}\n".format(name,seq)
        pass

