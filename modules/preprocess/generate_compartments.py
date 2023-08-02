import os
import sys
from multiprocessing import Process
from modules.utils import *


def generate_compartments_each(juicer_tools_path, hic_file,method, chrom, flag, bin_size, output_file):
    command = "java -jar {} eigenvector \
        {} {} \
        {} {} {} {}". \
        format(juicer_tools_path, method, hic_file, chrom, flag, bin_size, output_file)
    status = os.system(command)
    print("status:{}".format(status))
    if status != 0:
        sys.exit(-1)
        # raise Exception(f"Command '{command}' failed with exit status {status}")
        # sys.exit(-1)


class GenerateCompartments:
    def __init__(self, juicer_tools_path, hic_file, threads,chrom_size_file, method, flag,
                 bin_size, output_dir,output_prefix,verbose):

        self.juicer_tools_path, self.threads, self.method, self.flag, self.bin_size, self.output_dir = \
            juicer_tools_path, threads,method, flag, bin_size, output_dir
        self.hic_file = hic_file
        self.chrom_size_file = chrom_size_file
        self.output_prefix, self.verbose = output_prefix,verbose
        pass

    def generate_compartments(self,juicer_tools_path, hic_file, threads, chroms, method, flag, bin_size, output_dir,
                              output_prefix="eigen_", verbose=True):
        thread_pool_list = []
        records_tasks = initialize_all_tasks(len(chroms), threads)
        # thread_pool_list.append()
        # tasks = listen_batch(thread_pool_list)

        for index, task in enumerate(records_tasks):
            # execute the gene assign for each task
            task_id = index
            chrom_list = chroms[index]
            output_file_format = os.path.join(output_dir,
                                       output_prefix + "{}_{}.txt")
            if verbose:
                fprint(msg="Starting the {} epoch...".format(index))
            for index_job, item in enumerate(task):
                output_file = output_file_format.format(chroms[item], human_read_length(bin_size))
                p = Process(target=generate_compartments_each,
                            args=(juicer_tools_path, hic_file, method, chrom_list, flag, bin_size, output_file,))
                p.daemon = True
                # atexit.register(exit_handler,p)
                p.start()
                thread_pool_list.append(p)

        while True:
            complete = listen_batch(thread_pool_list)
            # complete = True
            if complete:
                # fprint('LOG', "Starting combining all temp gff files")
                # combine_all_temp_output(output, temp_dir, temp_gff_pattern, verbose)
                fprint('LOG', "Complete done!")
                # complete_number += 1
                # if verbose:
                # fprint('LOG',"Complete count: {}".format(complete_number))
                # fprint('LOG',"Done for the {} epoch".format(index+1))
                # print("{}: Done for the {} epoch......" \
                #       .format(time.asctime(), index + 1))
                break
            else:
                sys.exit(-1)

    def run(self):
        self.chroms = read_chrom_size(self.chrom_size_file)
        self.generate_compartments(self.juicer_tools_path,self.hic_file, self.threads, self.chroms, self.method,
                                   self.flag, self.bin_size, self.output_dir,self.output_prefix, verbose=self.verbose)

