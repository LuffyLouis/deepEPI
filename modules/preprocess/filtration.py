# import os
# import sqlite3
# import pandas as pd

from modules.utils import *
from modules.preprocess.common import ChunkReadAndRunThread


# import gffutils

class FilterInteractions(ChunkReadAndRunThread):
    """
    Filtration of interactions by distance, genomic annotation, annotation number, interaction type, length

    """
    def __init__(self, input_interaction, interact_type, promoter, distance, within_compartment, threads, chunksize,chrom_size_file,
                 annotation, promoter_range, promoter_number, distance_type, length, output_file, temp_dir, output_raw,
                 verbose):
        """

        :param input_interaction:
        :param interact_type:
        :param promoter:
        :param distance:
        :param within_compartment:
        :param threads:
        :param chunksize:
        :param chrom_size_file:
        :param annotation:
        :param promoter_range:
        :param promoter_number:
        :param distance_type:
        :param length:
        :param output_file:
        :param temp_dir:
        :param output_raw:
        :param verbose:
        """
        # super().__init__(input_interaction)
        super().__init__()
        self.input_interaction, self.interact_type, self.promoter, self.distance, self.within_compartment, self.threads, self.chunk_size, self.verbose = \
            input_interaction, interact_type, promoter, distance, within_compartment, threads, chunksize, verbose
        self.chrom_size_file = chrom_size_file
        self.output_file = output_file
        self.temp_dir = temp_dir
        self.temp_output_dir, self.temp_output_pattern = os.path.join(os.path.dirname(output_file),
                                                                      "temp_output"), "filt_interactions_\d+.txt"
        check_and_create_dir(self.temp_output_dir)
        self.annotation = annotation
        self.promoter_range = promoter_range
        self.promoter_number = promoter_number
        self.distance_type = distance_type
        self.length = length
        self.output_raw = output_raw
        self.temp_output_raw_dir = os.path.join(os.path.dirname(output_raw), "temp_output_raw")
        check_and_create_dir(self.temp_output_raw_dir)
        self.timer = Timer()
        self.temp_output_raw_pattern = "raw_interactions_\d+_\d+.txt"
        self.temp_output_raw_format = "raw_interactions_{}_{}.txt"
        self.sqlite_table = "genome_annotation"
        self.promoter_table = "promoter_{}".format(human_read_length(self.promoter_range))
        self.colnames = ["reads_name", "chrom1", "start1", "strand1", "chrom2", "start2", "strand2", "length",
                         "bin_name1", "bin_name2", "bin_index1", "bin_index2", "Unknown",
                         "compartment1", "compartment2", "compartment1_region", "compartment2_region", "end1", "end2"]

    def create_annotation_db(self):
        """

        """
        self.gff_db = read_gff(self.annotation, self.temp_dir)

        # self.gff_db.conn.close()

    def create_promoter_db(self, feature=None, attr=None, format="bed", annotation_format="gtf"):
        """

        :param feature:
        :param attr:
        :param format:
        :param annotation_format:
        """
        # self.promoter_table = {
        #     "chrom":[],
        #     "start":[],
        #     ""
        # }
        ## read the chrom size info from file
        self.chrom_size_info = read_chrom_size(self.chrom_size_file,return_type="raw")
        if attr == None:
            if annotation_format == "gff":
                attr = "gene_name"
            else:
                attr = "gene_id"
        if feature == None:
            if annotation_format == "gff":
                feature = "gene"
            else:
                feature = "gene"

        self.promoter_file = os.path.join(self.temp_dir, "annotation", "promoters.bed")
        if format == "bed":
            minus = 1
        else:
            minus = 0
        header = ["chrom", "start", "end", "name", "strand"]
        # print(self.chrom_size_info)
        for index, record in enumerate(self.gff_db.features_of_type(feature)):
            tss = record.start if record.strand == '+' else record.end
            # upstream_start = tss - 1000
            # upstream_end = tss - 1
            # downstream_start = tss + 1
            # downstream_end = tss + 1000
            chrom = record.chrom
            start = max(0,tss - minus - 1000)
            # print(chrom)

            # print(self.chrom_size_info.loc[self.chrom_size_info.iloc[:,0] == chrom,1])
            # print(sum(self.chrom_size_info.iloc[:, 0] == chrom))
            if len(self.chrom_size_info.loc[self.chrom_size_info.iloc[:,0] == chrom,1]) >= 1:

                end = min(tss + 999, self.chrom_size_info.loc[self.chrom_size_info.iloc[:,0] == chrom,1].item())
            else:
                fprint("WARNING","No found the size of {}, so the end position was set to Inf".format(chrom))
                end = tss + 999
            if attr in record.attributes.keys():
                name = "promoter_{}|{}".format(index, record.attributes[attr][0])
            else:
                name = "promoter_{}".format(index)
            strand = record.strand
            record_list = [chrom, start, end, name, strand]

            each_line = "\t".join([str(i) for i in record_list])
            if index == 0:
                self.output_func(self.promoter_file, "", init=True)
                self.output_func(self.promoter_file, "\t".join(header), init=False)
                self.output_func(self.promoter_file, each_line, init=False)
            else:
                self.output_func(self.promoter_file, each_line, init=False)
        # return
        # self.db_manager = DataFrameToSQLite(self.combined_eigen_db_path)
        # self.db_manager.create_database()
        # if not self.db_manager.is_table_exist(self.sqlite_table):
        #     self.db_manager.create_table(self.sqlite_table, combined_eigen_data)

        pass

    def query_within_promoter(self, chrom, start, end):
        """

        :param chrom:
        :param start:
        :param end:
        :return:
        """
        condition = "chrom = '{}' AND start <= '{}' AND end >= '{}'".format(chrom, start, end)
        res = self.promoter_db_manager.find_data(self.promoter_table, condition)
        if res.shape[0] >= 1:
            return {
                "within": 1,
                "count": res.shape[0],
                "res": res
            }
        else:
            return {
                "within": 0,
                "count": res.shape[0],
                "res": res
            }

    def get_within_promoter(self, res1, res2):
        promoter_info = []
        if res1["within"] + res2["within"] >= 1:
            if res1["res"].shape[0] >= 1:
                promoter_info.append(res1["res"].loc[0, "name"])
            # else:
            #     promoter_info.append("")
            if res2["res"].shape[0] >= 1:
                promoter_info.append(res2["res"].loc[0, "name"])
            # else:
            #     promoter_info.append("")

            return ",".join([str(i) for i in [res1["within"], res2["within"]]]), \
                   ",".join(promoter_info)
        else:
            return None, None

    def get_distance(self, ):
        pass

    def get_within_compartment(self, list1, list2):
        return [x and y for x, y in zip(list1, list2)]

    def filt_interactions(self, chunk_id, task_id, temp_data, temp_file):
        if chunk_id == 0 and task_id == 0:
            init = True
        else:
            init = False

        # temp_data = temp_data.copy()
        temp_data["interaction_type"] = None
        temp_data["promoter"] = None
        temp_data["promoter_info"] = None
        temp_data["distance"] = None
        temp_data["within_compartment"] = None
        temp_data["filter"] = False
        ## add annotation info
        # if return_annotation:
        temp_data.loc[temp_data["chrom1"] != temp_data["chrom2"], "interaction_type"] = "inter"
        temp_data.loc[temp_data["chrom1"] == temp_data["chrom2"], "interaction_type"] = "intra"

        # print(temp_data["start1"])
        temp_data["start1"] = temp_data["start1"].astype(int)
        temp_data["end1"] = temp_data["end1"].astype(int)
        temp_data["start2"] = temp_data["start2"].astype(int)
        temp_data["end2"] = temp_data["end2"].astype(int)

        for i in range(temp_data.shape[0]):
            if self.promoter:
                res1 = self.query_within_promoter(temp_data.loc[i, "chrom1"], temp_data.loc[i, "start1"],
                                                  temp_data.loc[i, "end1"])
                res2 = self.query_within_promoter(temp_data.loc[i, "chrom2"], temp_data.loc[i, "start2"],
                                                  temp_data.loc[i, "end2"])
                promoter, promoter_info = self.get_within_promoter(res1, res2)
                temp_data.loc[i, "promoter"] = promoter
                temp_data.loc[i, "promoter_info"] = promoter_info
            ##
            if temp_data.loc[i, "interaction_type"] == "intra":
                # intra_index = temp_data["interaction_type"] == "intra"
                dis1 = abs(temp_data.loc[i, "start1"] - temp_data.loc[i, "end2"])
                dis2 = abs(temp_data.loc[i, "start2"] - temp_data.loc[i, "end1"])
                max_distance = max(dis1, dis2)
                min_distance = min(dis1, dis2)
                mean_distance = min_distance + (max_distance - min_distance) / 2
                if self.distance_type.lower() == "longer":
                    temp_data.loc[i, "distance"] = max_distance
                elif self.distance_type.lower() == "shorter":
                    temp_data.loc[i, "distance"] = min_distance
                else:
                    temp_data.loc[i, "distance"] = mean_distance
            ##
            # print(type(temp_data.loc[i, ["compartment1", "compartment2"]]!= "Multi-compartment"))
            within_res = self.get_within_compartment(list(temp_data.loc[i, ["compartment1", "compartment2"]].notna()),
                                                     list(temp_data.loc[i, ["compartment1",
                                                                            "compartment2"]] != "Multi-compartment"))
            temp_data.loc[i, "within_compartment"] = sum(within_res)



        temp_data_filt = temp_data.copy()
        ## filter chromosome interaction types
        if self.interact_type:
            if self.interact_type.lower() == "inter":
                temp_data_filt = temp_data_filt.loc[temp_data_filt["chrom1"] != temp_data_filt["chrom2"], :]
            elif self.interact_type.lower() == "intra":
                temp_data_filt = temp_data_filt.loc[temp_data_filt["chrom1"] == temp_data_filt["chrom2"], :]
            else:
                # temp_data_filt = temp_data_filt
                fprint("WARNING",
                       "No valid interaction type keyword, so it won't filter the interactions based on their types")
                pass
            # pass
        if self.promoter:
            # self.query_within_promoter()
            temp_data_filt = temp_data_filt.loc[temp_data_filt["promoter"].notna(), :]
            temp_data_filt = temp_data_filt.loc[
                             temp_data_filt["promoter"].apply(self.get_within_promoter_num) == self.promoter_number, :]
            pass
        if self.distance:
            temp_data_filt = temp_data_filt.loc[temp_data_filt["distance"] <= self.distance, :]
            pass
        if self.within_compartment:
            temp_data_filt = temp_data_filt.loc[temp_data_filt["within_compartment"] == self.within_compartment, :]
            pass
        if self.length is not None:
            temp_data_filt = temp_data_filt.loc[temp_data_filt["length"] <= self.length, :]

        temp_data_filt["filter"] = True
        temp_data.loc[temp_data_filt.index,"filter"] = True
        temp_data_filt = temp_data_filt.reset_index(drop=True)

        ## write the raw interactions before filtration
        if self.output_raw:
            # if chunk_id ==
            self.write_output(temp_data, os.path.join(self.temp_output_raw_dir,
                                                      self.temp_output_raw_format.format(chunk_id, task_id)), init=init)
        self.write_output(temp_data_filt, temp_file, format="txt", init=init)
        pass

    def get_within_promoter_num(self, promoter):
        return sum([int(i) for i in promoter.split(",")])

    # def
    def run(self):
        self.timer.start()
        ##
        if self.promoter:
            self.create_annotation_db()
            self.create_promoter_db()
            promoter_data = pd.read_table(self.promoter_file, sep="\t")
            # promoter_data
            ##
            self.combined_eigen_db_path = os.path.join(self.temp_dir, "local.db")
            self.promoter_db_manager = DataFrameToSQLite(self.combined_eigen_db_path)
            self.promoter_db_manager.create_database()
            if not self.promoter_db_manager.is_table_exist(self.promoter_table):
                self.promoter_db_manager.create_table(self.promoter_table, promoter_data)
        ## read interactions by chunk size
        self.valid_pairs_data_iterator = self.read_valid_pairs(self.input_interaction, self.chunk_size, self.colnames,
                                                               header=True)
        for chunk_id, chunk_data in enumerate(self.valid_pairs_data_iterator):
            fprint(msg="Starting the chunk {}...".format(chunk_id))
            self.run_each(chunk_id, chunk_data, self.threads, self.output_file, self.temp_output_dir,
                          self.temp_output_pattern,
                          fun_thread=self.filt_interactions, fun_args=(),
                          verbose=self.verbose)
            ## combine and write the raw interactions
            self.combine_interactions(chunk_id, self.output_raw, self.temp_output_raw_dir, self.temp_output_raw_pattern,
                                      self.verbose)

        self.timer.stop()
        fprint(msg="Complete done! Elapsed time: {:.3f}s".format(self.timer.elapsed_time()))
        ## close the promoter db manager
        if self.promoter:
            self.promoter_db_manager.close_connection()
