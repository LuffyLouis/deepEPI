# -*- coding: utf-8
import os
import sys
import time

import gffutils
import pandas as pd
from Bio import SeqFeature


def fprint(type="LOG", msg=''):
    print("{}|[{}]: {}" \
          .format(time.asctime(), type, msg))
    if type.lower() == "error":
        sys.exit(-1)

def fprint_step(module=None,step=None):
    if module:
        title_len = len(module)
        print("#" * 90)
        print("#" * 5 + " " * int((90-title_len-5*2)/2) + "{}".format(module) + " " * int((90-title_len-5*2)/2) + "#" * 5)
        print("#" * 90)
    if step:
        subtitle_len = len(step)
        print("#" * int((90-subtitle_len-5*2)/2) + " " * 5 + "{}".format(step) + " " * 5 + "#" * int((90-subtitle_len-5*2)/2))

# check the exist of temp_gff_dir
def check_and_create_dir(temp_gff_dir,overwrite=False):
    if os.path.exists(temp_gff_dir):
        if overwrite:
            fprint("WARNING","!!Find the temporary directory has been exist and the temp file will be overwrite!!")
            # os.rmdir(temp_gff_dir)
            os.system("rm -rf {}/*".format(temp_gff_dir))
        # os.mkdir(temp_gff_dir)
        else:
            fprint("WARNING","!!Find the temporary directory has been exist and the temp file will be reused!!")
        pass
    else:
        fprint(msg="Not find the temporary directory: {}, now create it!".format(temp_gff_dir))
        os.makedirs(temp_gff_dir)


def read_chrom_size(chrom_size_file, return_type="chrom"):
    chrom_size_data = pd.read_table(chrom_size_file,header=None,sep="\t")
    if return_type == "chrom":

        return chrom_size_data.iloc[:, 0]
    # elif return_type == "raw":
    #     return chrom_size_data
    else:
        return chrom_size_data

def format_location(location_list,strand=None):
    if len(location_list) == 3:
        if strand:
            return "{}:{}-{}/{}".format(location_list[0],location_list[1],location_list[2],strand)

        return "{}:{}-{}".format(location_list[0],location_list[1],location_list[2])
    elif len(location_list) == 4:
        return "{}:{}-{}/{}".format(location_list[0], location_list[1], location_list[2], location_list[3])

    return ""

def mean(object):
    if type(object) is list or type(object) is tuple:
        return sum(object)/len(object)
    else:
        raise "Not supported data type"
    pass
# 自定义参数类型转换函数
def auto_type(s):
    if s.isdigit():  # 判断是否为数字
        return int(s)  # 如果是数字，则转换为整数类型
    else:
        return s  # 否则返回字符串类型

def read_list_file(list_file,sep="\t"):
    list_data = pd.read_table(list_file,header=None,sep=sep)
    return list_data.iloc[:,0].to_list()
    pass


def filt_chrom(chrom_size_data, keep_chrom_file=None, discard_chrom_file=None):
    chrom_size_data_filt = chrom_size_data.copy()
    if keep_chrom_file:
        keep_chroms = read_list_file(keep_chrom_file)
        chrom_size_data_filt = chrom_size_data_filt.loc[chrom_size_data_filt.iloc[:,0].isin(keep_chroms),:]
    if discard_chrom_file:
        discard_chroms = read_list_file(discard_chrom_file)
        # keep_chroms = read_list_file(keep_chrom_file)
        chrom_size_data_filt = chrom_size_data_filt.loc[~chrom_size_data_filt.iloc[:,0].isin(discard_chroms), :]
    return chrom_size_data_filt


def human_read_length(raw_length,reverse=False):
    units = ['', 'k', 'M', 'G', 'T', 'P', 'E']  # 单位列表
    res = None
    if reverse:
        if 'k' in raw_length.lower():
            res = float(raw_length.split("k")[0]) * 1e3
        elif 'M' in raw_length.upper():
            res = float(raw_length.split("M")[0]) * 1e6
        elif 'M' in raw_length.upper():
            res = float(raw_length.split("M")[0]) * 1e9
        elif 'T' in raw_length.upper():
            res = float(raw_length.split("T")[0]) * 1e12
        elif 'P' in raw_length.upper():
            res = float(raw_length.split("P")[0]) * 1e15
        elif 'E' in raw_length.upper():
            res = float(raw_length.split("E")[0]) * 1e18
        return res
    else:
        raw_length = int(raw_length)
        # units = ['', 'k', 'M', 'G', 'T', 'P', 'E']  # 单位列表
        unit_index = 0

        while abs(raw_length) >= 1000 and unit_index < len(units) - 1:
            raw_length /= 1000
            unit_index += 1

        return '{:.0f}{}b'.format(raw_length, units[unit_index])


# To check whether the thread pool is complete
def listen_batch(thread_pool_list):
    incomplete = False
    for thread in thread_pool_list:
        incomplete += thread.is_alive()
        exitcode = thread.exitcode
        if exitcode != 0:
            return None
    if incomplete >= 1:
        return False
    else:
        return True


def batch_jobs(tasks, batch_size):
    """
    :param tasks, list
    :param batch_size, size of each batch
    return batch_tasks, list in list
    """
    count = 0
    batch_tasks = []
    item_task = []
    for item in tasks:
        item_task.append(item)
        count += 1
        if count % batch_size == 0:
            if count > 0:
                batch_tasks.append(item_task)
            item_task = []

    if count % batch_size != 0:
        batch_tasks.append(item_task)

    return batch_tasks


def initialize_all_tasks(total_count, jobs):
    # records_tasks = []
    # total_count = data.shape[0]
    batch_size = max(int(total_count / jobs), 1)
    records_tasks = batch_jobs(range(total_count), batch_size)
    return records_tasks


def get_intersection(chrom1, start1, end1, chrom2, start2, end2):
    # 创建第一个特征对象
    feature1 = SeqFeature.FeatureLocation(start=start1, end=end1, strand=1)

    # 创建第二个特征对象
    feature2 = SeqFeature.FeatureLocation(start=start2, end=end2, strand=1)

    # 计算交集
    intersection = feature1.intersection(feature2)

    # 返回结果
    return {
        'chromosome': chrom1,
        'start_position': intersection.start,
        'end_position': intersection.end
    }

### GFF/GTF3
def create_gff_db(input_annotation,database_file_path):
    if database_file_path is None:
        check_res = False
    else:
        check_res = check_gff_db(database_file_path)
    if not check_res:
        suffix = os.path.basename(input_annotation).split(".")[-1]
        if suffix.lower() == "gtf":
            db = gffutils.create_db(input_annotation, database_file_path,force=False,disable_infer_transcripts=True,
                                merge_strategy="create_unique")
        else:
            db = gffutils.create_db(input_annotation, database_file_path, force=False, disable_infer_transcripts=True,
                                    disable_infer_genes=True,
                                    merge_strategy="create_unique")
    else:
        db = gffutils.FeatureDB(database_file_path)
    # db = gffutils.create_db(input_annotation, database_filename, force=True,
    #                         merge_strategy="create_unique")
    return db


def check_gff_db(database_filename):
    if os.path.exists(database_filename):
        return True
    else:
        return False
    pass

def read_gff(annotation,temp_dir,database_file_path=None):
    temp_gff_dir = os.path.join(temp_dir,"annotation")
    annotation_suffix = os.path.splitext(annotation)[1]
    check_and_create_dir(temp_gff_dir)
    if database_file_path:
        # database_file_path = database_file_path
        pass
    else:
        database_file_path = os.path.join(temp_gff_dir,annotation.split(annotation_suffix)[0] + ".sqlite3")
    print(database_file_path)
    gff_db = create_gff_db(annotation,database_file_path)
    return gff_db