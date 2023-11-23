# -*- coding: utf-8

# from multiprocessing import Process
import argparse
import traceback
import sys
# import os
from modules.datasets.prep_datasets import PrepDatasets
from modules.models.training_model import TrainModel
from modules.predict.predict import PredictEPI
from modules.preprocess import generate_compartments, generate_interactions, filtration, balance, extract_seqs
from modules.logger import progress_logger
from modules.utils import *
# from modules.utils.tools import print_pixel_title
#
# print("""
# ██████╗ ███████╗███████╗██████╗ ███████╗██████╗ ██╗
# ██╔══██╗██╔════╝██╔════╝██╔══██╗██╔════╝██╔══██╗██║
# ██║  ██║█████╗  █████╗  ██████╔╝█████╗  ██████╔╝██║
# ██║  ██║██╔══╝  ██╔══╝  ██╔═══╝ ██╔══╝  ██╔═══╝ ██║
# ██████╔╝███████╗███████╗██║     ███████╗██║     ██║
# ╚═════╝ ╚══════╝╚══════╝╚═╝     ╚══════╝╚═╝     ╚═╝
# """)
parser = argparse.ArgumentParser(description="""
██████╗ ███████╗███████╗██████╗ ███████╗██████╗ ██╗
██╔══██╗██╔════╝██╔════╝██╔══██╗██╔════╝██╔══██╗██║
██║  ██║█████╗  █████╗  ██████╔╝█████╗  ██████╔╝██║
██║  ██║██╔══╝  ██╔══╝  ██╔═══╝ ██╔══╝  ██╔═══╝ ██║
██████╔╝███████╗███████╗██║     ███████╗██║     ██║
╚═════╝ ╚══════╝╚══════╝╚═╝     ╚══════╝╚═╝     ╚═╝
""",
                                 formatter_class=argparse.RawTextHelpFormatter)
# print_pixel_title("Welcome to My Program")
# parser.description =

# parser.add_argument("-h", "--help", action="help", help="查看帮助信息")

# parser.usage
subparsers = parser.add_subparsers(title="Module",dest="module",help="Run specific module to run. The following modules are provided (note: if the main module was specified, all steps in this main module will be implemented):\n"
                       "Preprocess: Preprocess the interactions generated from HiC-Pro\n"
                       "    generate_interactions: generate valid interactions within compartment A/B\n"
                       "    filtration: filter the valid interactions based on user specified parameters\n"
                       "    balance: balance the dataset with positive and negative labels\n"
                       "    extract_seqs: extract genomic sequences according to the genomic coordinates of interactions\n"
                       "Datasets: Split and generate the standard dataset format\n"
                       "    split_datasets: split the dataset into different parts, including training, validation, and test dataset\n"
                       "Train: Train the datasets with a specific model\n"
                       "Evaluation: Evaluate the performance under the current trained model with some metrics\n"
                       "Predict: Predict the Enhancer-Promoter interaction with a specifc model", metavar="MODULE")

# parser.add_argument("-m", "--module", dest="module",default="all",
#                   help="Run specific module to run. The following modules are provided (note: if the main module was specified, all steps in this main module will be implemented):\n"
#                        "Preprocess: \n"
#                        "    generate_interactions: generate valid interactions within compartment A/B\n"
#                        "    filtration: filt the valid interactions based on user specified parameters\n"
#                        "    balance: \n"
#                        "    extract_seqs\n"
#                        "Datasets:\n"
#                        "    split_datasets:\n"
#                        "Train:\n"
#                        "Evaluation:\n"
#                        "Prediction:", metavar="STRING")
preprocess_module = subparsers.add_parser('Preprocess',formatter_class=argparse.RawTextHelpFormatter)
preprocess_module.add_argument("-s","--step",dest="step",default=None,help="choose specific step to run\n"
                                           "    generate_interactions: generate valid interactions within compartment A/B\n"
                                           "    filtration: filt the valid interactions based on user specified parameters\n"
                                           "    balance: \n"
                                           "    extract_seqs\n")
# preprocess_group = parser.add_argument_group("Preprocess")
preprocess_module.add_argument("--input",help="the input file")
preprocess_module.add_argument("--temp_dir",help="the temporary directory")
preprocess_module.add_argument("--output",help="the output file")
preprocess_module.add_argument("--threads",type=int,default=1,help="threads (default=1)\n"
                                                                   "Note: it is not built for predict mode!!!")
preprocess_module.add_argument("--chunksize",type=float,default=1e5,help="chunksize (default=1e5)")
preprocess_module.add_argument("--verbose",action="store_true", default=False, help="whether to open the verbose mode\n"
                                                                                    "default=False")

preprocess_group = preprocess_module.add_argument_group("generate_compartments")
preprocess_group.add_argument("--hic_file",help="the .hic file which can be generated from HiC-Pro")
preprocess_group.add_argument("--juicer_tools_path",help="the absolute or relative path of juicer tools binary file")

preprocess_group.add_argument("--chrom_size_file",help="the chrom size for reference genome")
preprocess_group.add_argument("--method",help="the method in juicer_tools eigenvector to calculating eigenvector\n"
                                              "<NONE/VC/VC_SQRT/KR>")
preprocess_group.add_argument("--flag",help="the position flag in juicer_tools eigenvector\n"
                                            "<BP/FRAG>")
preprocess_group.add_argument("--bin_size",type=int,help="the bin size in juicer_tools eigenvector")
preprocess_group.add_argument("--output_dir",help="output directory path")
preprocess_group.add_argument("--output_prefix",default="eigen_",help="output file prefix\n"
                                                                      "default=eigen_")


preprocess_group = preprocess_module.add_argument_group("generate_interactions")
# preprocess_group.add_argument("--valid_pairs_file",help="the .allValidPairs file which is generated from HiC-Pro")

preprocess_group.add_argument("--eigenvec_dir",help="the eigenvector directory containing eigenvector calculated from juicer_tools\n"
                                                    "It generally includes all similar eigen files at a same resolution of different chromosomes/scaffolds")
preprocess_group.add_argument("--eigenvec_file_pattern",default="eigen_(.*)_(.*).txt",help="the RegExp pattern for these eigenvector files and it must conform the following format\n"
                                                             "e.g. eigen_(.*)_(.*).txt -- eigen_chr1_500kb.txt\n"
                                                             "default='eigen_(.*)_(.*).txt'")
# preprocess_group.add_argument("--bin_size",type=int,help="the bin size or resolution you specified")
preprocess_group.add_argument("--keep_chrom_file",default=None,help="which chromosomes you want to keep\n"
                                                                    "default=None")
preprocess_group.add_argument("--discard_chrom_file",default=None,help="which chromosomes you want to discard\n"
                                                                       "default=None")

preprocess_group = preprocess_module.add_argument_group("filtration")
# preprocess_group.add_argument("--input_interaction", help="the input interaction file")
preprocess_group.add_argument("--interact_type", default="intra",help="the interaction type filter\n"
                                                      "parameters: intra, inter, all\n"
                                                      "default='intra'")
preprocess_group.add_argument("-a", "--annotation", dest="annotation", help="Corresponding gene annotation gtf file")
preprocess_group.add_argument("--promoter", action="store_true",default=False, help="whether to filter the interactions containing promoter region\n"
                                                                                     "default=False")
preprocess_group.add_argument("--promoter_number", type=int, default=1, help="whether to filter the interactions containing promoter region\n"
                                                                                     "default=1")
preprocess_group.add_argument("--promoter_range", type=float, default=1e3,help="the length of promoter flanking region\n"
                                                                                     "default=1e3")
preprocess_group.add_argument("--output_raw", default=True,help="whether to save raw annotated interactions before filtration\n"
                                                                                     "default=True")
preprocess_group.add_argument("--length", type=float, default=None, help="the maximum length of interaction\n"
                                                                          "default=None")
preprocess_group.add_argument("--distance", type=float, default=1e6, help="the maximum distance between two regions for each interaction\n"
                                                                          "default=1e6 (1Mb)")
preprocess_group.add_argument("--distance_type", default="median", help="how to calculate the distance between two interactions, and optional distance types\n"
                                                                          "include longer, median, shorter, default=median\n"
                                                                        "   longer: larger end - smaller start position\n"
                                                                        "   median: use the mid points\n"
                                                                        "   shorter: larger start - smaller end position")
preprocess_group.add_argument("--within_compartment", type=int,default=2, help="whether to filter the interactions both within a compartment but not across multi compartments\n"
                                                                                "default=2")

preprocess_group = preprocess_module.add_argument_group("balance")
preprocess_group.add_argument("--raw_interactions_file", help="the raw interaction file generated from above filtration step")
preprocess_group.add_argument("--reference_genome", dest="reference_genome", help="the reference genome which must accordant with the annotation and other input files")
preprocess_group.add_argument("--GC_content", default=0.5,help="whether to balance dataset with most matched GC content (boolen)\n"
                                                                "and you can also set the specific number to discard any unvalid negative\n"
                                                                "interactions with larger GC content difference than this value you given\n"
                                                               "default=0.5")
preprocess_group.add_argument("--balanced_interaction_type", default='intra', help="to balance positive dataset with matched negative dataset"
                                    "with which interaction type\n"
                                   "parameters: intra, inter, all\n"
                                   "default='intra'")
preprocess_group.add_argument("--balanced_same_chromsome", default=True, help="whether to balance positive dataset with matched negative dataset"
                                    "with same chromosome\n"
                                  "Note: this parameter is only valid when the --balanced_interaction_type is intra!"
                                   "default=True")
preprocess_group = preprocess_module.add_argument_group("extract_dataset")
preprocess_group.add_argument("--raw_interaction_fasta_file",help="the fasta file for raw interactions")
preprocess_group.add_argument("--dataset_name",default="raw", help="the encoded seqs dataset name in .h5 file\n"
                                                     "default=raw")
preprocess_group.add_argument("--enhancer_length",type=float,default=4e3, help="the length to adjust the raw enhancer sequence by padding or triming strategy\n"
                                                     "default=2e3")
preprocess_group.add_argument("--promoter_length",type=float,default=2e3, help="the length to adjust the raw promoter sequence by padding or triming strategy\n"
                                                     "default=2e3")
preprocess_group.add_argument("--trim",default="both", help="the triming strategy\n"
                                                            "optional parameters: left, right, both\n"
                                                     "default=both")
preprocess_group.add_argument("--padding",default="both", help="the padding strategy\n"
                                                               "optional parameters: left, right, both\n"
                                                     "default=both")
preprocess_group.add_argument("--compression",default="gzip", type=auto_type, help="(String or int) Compression strategy. \n"
                                                                                   "Legal values are 'gzip', 'szip', 'lzf'. If an integer in range(10),this indicates gzip compression level. \n"
                                                                                   "Otherwise, an integer indicates the number of a dynamically loaded compression filter.\n"
                                                                                   "Compression level will be more efficient, but the speed will be slower when the level number is larger\n"
                                                                 "default=gzip")
preprocess_group.add_argument("--pretrained_vec_file", help="the pretrained dna2vec file when the encode_method was set to dna2vec")
preprocess_group.add_argument("--k_mer", default=6, help="the k value of k-mer or k-list specified when the encode_method was set to dna2vec\n"
                                                         "multiple k-mer must be separated by comma, e.g. 4,5,6\n"
                                                         "default=6, which means 6-mer")
# pretrained_vec_file,k_mer,
# preprocess_group.add_argument("--promoter_length",type=float,default=, help="the strategy to pad the incomplete sequence\n"
#                                                      "default=raw")
preprocess_group.add_argument("--concat_reverse",action="store_true",default=False,help="whether to concat the interaction sequences with reverse strand\n"
                                                                   )
preprocess_group.add_argument("--concat_epi", action="store_true",default=False,help="whether to concat the enhancer and promoter together\n"
                                                               "If not, the {dataset_name}_seq1 and {dataset_name}_seq2 will be generated separately\n"
                                                               )
preprocess_group.add_argument("--encode_method",default="dna2vec", help="the encode method to encode raw sequences\n"
                                                                         "optional parameters: onehot, dna2vec\n"
                                                                        "default=True"
                                                    )
datasets_module = subparsers.add_parser('Datasets',formatter_class=argparse.RawTextHelpFormatter)
datasets_module.add_argument("-i", "--input",help="input file")
datasets_module.add_argument("-s", "--step",help="the specific step to run")
datasets_module.add_argument("--split",default="8:2",help="the training:validation:test ratio\n"
                                            "the split ratio must be separated by colon\n"
                                            "e.g. 8:2 (default)")
datasets_module.add_argument("--key",help="the specific step to run")
datasets_module.add_argument("--chunksize",type=float,default=1e3,help="chunksize (default=1e5)")
datasets_module.add_argument("--test_size",type=float,default=0.2,help="the specific step to run")
datasets_module.add_argument("--output_format",default="torch",help="(deprecated) the specific step to run")
datasets_module.add_argument("--random_state",type=int,default=0,help="the random state to randomly split raw datasets\n"
                                                            "default=0")
datasets_module.add_argument("--output_prefix",help="the output file path and filename prefix")
datasets_module.add_argument("--compression",type=auto_type,default="gzip",help="the compression method or level \n"
                                                                                "(the detailed description can be found in the help page of Preprocess module)\n"
                                                                                "default: gzip")

# datasets_module.add_argument("", help="")

datasets_module.add_argument("--verbose", action="store_true", default=False, help="whether to open the verbose mode")
train_module = subparsers.add_parser("Train",formatter_class=argparse.RawTextHelpFormatter)
train_module.add_argument('--nodes', default=1,type=int, metavar='N')
train_module.add_argument('--gpus', default=1,type=int,help='number of gpus per node')
train_module.add_argument('--nr', default=0, type=int,help='ranking within the nodes')
train_module.add_argument('--master', default=None,help='the address of master node (IP:PORT)\n'
                                                                 'e.g. 192.168.10.1:8888')
train_module.add_argument("-i","--input",help="input .h5 file for training")
train_module.add_argument("--test_input",default=None,help="input .h5 file for test")

train_module.add_argument("--train_dataset_dir",help="the directory for .h5 files")
train_module.add_argument("--train_dataset_pattern",help="the RegExp pattern of input .h5 files")
train_module.add_argument("--workers",type=int,default=0,help="the workers to load dataset")

train_module.add_argument("--model",help="the model selected to train\n"
                                         "optional: \n"
                                         "  simpleCNN\n"
                                         "  CNN\n"
                                         "  EPIMind")
train_module.add_argument("--encode_method",default="onehot",help="the RegExp pattern of input .h5 files")
train_module.add_argument("--concat_reverse",action="store_true",default=False,help="the RegExp pattern of input .h5 files")
#
train_module.add_argument("--enhancer_len",type=float,default=4e3,help="the max length of enhancer")
train_module.add_argument("--promoter_len",type=float,default=2e3,help="the max length of promoter")
train_module.add_argument("--heads",type=int,default=8,help="the number of heads in multi-head attention")
train_module.add_argument("--num_layers",type=int,default=4,help="the number of stacks of encoder")
train_module.add_argument("--num_hiddens",type=int,default=72,help="the number of elements in hidden layer")
train_module.add_argument("--ffn_num_hiddens",type=int,default=256,help="the number of elements in hidden layer in Feed forward hidden layer")
# enhancer_len, promoter_len, heads, num_layers, num_hiddens, ffn_num_hiddens
# encode_method,concat_reverse
train_module.add_argument("--is_param_optim",action="store_true",default=False,help="whether to optimize the hyper-parameters")
train_module.add_argument("--param_optim_strategy",help="the strategy for hyper-parameters optimization\n"
                                                        "optional:\n"
                                                        "   random: randomized parameters optimization\n"
                                                        "   grid: grid parameters optimization")
train_module.add_argument("--params_config",default="config.txt",help="the config file for hyper-parameters optimization")
train_module.add_argument("--random_size",type=int,default=12,help="the size of random hyper-parameters optimization in random parameters optimization strategy\n"
                                                                   "default=12")

train_module.add_argument("--init_points",type=int,default=5,help="the step of random search in bayes optimization\n"
                                                                   "default=5")
train_module.add_argument("--n_iter",type=int,default=10,help="the number of bayes iterations\n"
                                                                   "default=10")
train_module.add_argument("--k_fold",type=int,default=None,help="the number of k-fold cross validation (CV)")
train_module.add_argument("--metrics",default="auc",help="the average value of metric for evaluating this model in k fold CV\n"
                                                         "optional: \n"
                                                         "  auc: the area under curve for ROC curve\n"
                                                         "  f1: a systematic metric considering precision and recall\n"
                                                         "  acc: the accuracy at test dataset")
train_module.add_argument("--save_fig",default="fig.pdf",help="whether to save fig and corresponding save path:\n"
                                                              "default: fig.pdf")
train_module.add_argument("--init",default="",help="the initialization method for parameters in model\n"
                                                   "optional: \n"
                                                   "    zeros: all zeros initialization\n"
                                                   "    normal: initialization parameters conform to general normalization distribution\n"
                                                   "    uniform: initialization parameters conform to uniform distribution\n"
                                                   "    xavier_normal: Xavier normal initialization\n"
                                                   "    kaiming_normal: Kaiming normal initialization\n"
                                                   "    xavier_uniform: Xavier uniform initialization\n"
                                                   "    kaiming_uniform: Kaiming uniform initialization")
train_module.add_argument("--epochs",type=int,default=10,help="the epochs (default=10)")
train_module.add_argument("--lr",type=float,default=0.001,help="learning rate (default=0.001)")
train_module.add_argument("--optimizer",default="SGD",help="the optimizer to implement the parameters optimization\n"
                                                           "optional: \n"
                                                           "    SGD: stochastic gradient descent\n"
                                                           "    Adam: Adaptive Moment Estimation\n"
                                                           "    Adagrad: Adaptive gradient\n"
                                                           "    Adadelta: Adaptive delta (Hinton, 2012)\n"
                                                           "    RMSprop: Root mean square prop\n"
                                                           "    ASGD: Averaged Stochastic Gradient Descent\n"
                                                           "    LBFGS: Limited-memory BFGS\n"
                                                           "    ...")
train_module.add_argument("--batch_size",type=int,default=32,help="batch size (default=32)")
train_module.add_argument("--momentum",type=float,default=0.9,help="momentum in optimizer")
train_module.add_argument("--weight_decay",type=float,default=0.9,help="weight_decay in optimizer")
train_module.add_argument("--nesterov",action="store_true",default=False,help="nesterov vector in optimizer")
train_module.add_argument("--betas",type=tuple,default=(0.9, 0.999),help="betas in optimizer")
train_module.add_argument("--eps",type=float,default=1e-8,help="eps in optimizer")
train_module.add_argument("--lr_decay",type=float,default=0,help="lr_decay in optimizer")
train_module.add_argument("--initial_accumulator_value",type=float,default=0,help="initial_accumulator_value in optimizer")

train_module.add_argument("--rho",type=float,default=0.9,help="rho in optimizer")
train_module.add_argument("--alpha",type=float,default=0.99,help="alpha in optimizer")
train_module.add_argument("--lambd",type=float,default=0.0001,help="lambd in optimizer")
train_module.add_argument("--t0",type=float,default=1e6,help="t0 in optimizer")
train_module.add_argument("--max_iter",type=int,default=20,help="momentum in optimizer LBFGS")
train_module.add_argument("--max_eval",type=int,default=None,help="max_eval in optimizer LBFGS")
# weight_decay, nesterov, betas, eps, lr_decay, initial_accumulator_value, rho, alpha,
#                  lambd, t0, max_iter, max_eval,

train_module.add_argument("--device",default=None,help="the physical device to train the model\n"
                                                        "optional: cpu, cuda or other else supported in pytorch\n"
                                                        "default=None, which means it will use the cuda firstly if with Nvidia GPU, and if not, the cpu will be used")
train_module.add_argument("--random_state",type=int,default=0,help="the random state to randomly split raw datasets\n"
                                                            "default=0")
train_module.add_argument("--rerun",action="store_true",default=False,help="whether to rerun and if not, it would load the latest parameters in previous results")

train_module.add_argument("--save_param_dir",default="cache",help="whether to save parameters and if so, the corresponding saved directory")
train_module.add_argument("--save_param_prefix",default="weights",help="whether to save parameters and if so, the corresponding saved file prefix")
train_module.add_argument("--log_dir",default="log",help="the logger directory")

train_module.add_argument("--save_model",help="whether to save model and if so, rhe corresponding save path")
train_module.add_argument("--verbose",action="store_true", default=False, help="whether to open the verbose mode")


predict_module = subparsers.add_parser("Predict",formatter_class=argparse.RawTextHelpFormatter)
predict_module.add_argument("--threads",type=int,default=1,help="threads (default=1)")
predict_module.add_argument("--enhancer_file",help="input .h5 file for enhancer")
predict_module.add_argument("--promoter_file",help="input .h5 file for promoter")
predict_module.add_argument("-e","--enhancer_seq",help="enhancer_seq")
predict_module.add_argument("-p","--promoter_seq",help="promoter_seq")

predict_module.add_argument("--pretrained_vec_file", help="the pretrained dna2vec file when the encode_method was set to dna2vec")
predict_module.add_argument("--k_mer", default=6, help="the k value of k-mer or k-list specified when the encode_method was set to dna2vec\n"
                                                         "multiple k-mer must be separated by comma, e.g. 4,5,6\n"
                                                         "default=6, which means 6-mer")
predict_module.add_argument("--trim",default="both", help="the triming strategy\n"
                                                            "optional parameters: left, right, both\n"
                                                     "default=both")
predict_module.add_argument("--padding",default="both", help="the padding strategy\n"
                                                               "optional parameters: left, right, both\n"
                                                     "default=both")

predict_module.add_argument("--model", default="simpleCNN",help="the model selected to train\n"
                                                                "default=simpleCNN\n"
                                         "optional: \n"
                                         "  simpleCNN\n"
                                         "  CNN\n"
                                         "  EPIMind")
predict_module.add_argument("--encode_method",default="onehot",help="the RegExp pattern of input .h5 files")
predict_module.add_argument("--concat_reverse",action="store_true",default=False,help="the RegExp pattern of input .h5 files")
predict_module.add_argument("--concat_epi", action="store_true",default=False,help="whether to concat the enhancer and promoter together\n"
                                                               "If not, the {dataset_name}_seq1 and {dataset_name}_seq2 will be generated separately\n"
                                                               )
#
predict_module.add_argument("--enhancer_len",type=float,default=4e3,help="the max length of enhancer\n"
                                                                         "default=4e3")
predict_module.add_argument("--promoter_len",type=float,default=2e3,help="the max length of promoter\n"
                                                                         "default=2e3")
predict_module.add_argument("--heads",type=int,default=8,help="the number of heads in multi-head attention\n"
                                                              "default=8")
predict_module.add_argument("--num_layers",type=int,default=4,help="the number of stacks of encoder\n"
                                                                   "default=4")
predict_module.add_argument("--num_hiddens",type=int,default=72,help="the number of elements in hidden layer\n"
                                                                     "default=72")
predict_module.add_argument("--ffn_num_hiddens",type=int,default=256,help="the number of elements in hidden layer for Feed forward hidden layer\n"
                                                                          "default=256")


predict_module.add_argument("--device",default=None,help="the physical device to train the model\n"
                                                        "optional: cpu, cuda or other else supported in pytorch\n"
                                                        "default=None, which means it will use the cuda firstly if with Nvidia GPU, and if not, the cpu will be used")
predict_module.add_argument("--save_param_dir",default="cache",help="whether to save parameters and if so, the corresponding saved directory\n"
                                                                    "default=cache")
predict_module.add_argument("--save_param_prefix",default="weights",help="whether to save parameters and if so, the corresponding saved file prefix\n"
                                                                         "default=weights")
predict_module.add_argument("--verbose",action="store_true", default=False, help="whether to open the verbose mode\n"
                                                                                 "default=False")


#
# enhancer_file = arg.enhancer_file
#         promoter_file = arg.promoter_file
#         enhancer_seq,promoter_seq = arg.enhancer_seq, arg.promoter_seq
#         model = arg.model
#         concat_epi, trim, padding, k_mer, pretrained_vec_file, threads = arg.concat_epi,arg.trim, arg.padding,arg.k_mer,arg.pretrained_vec_file,arg.threads
#         # train_dataset_dir,train_dataset_pattern,model,is_param_optim = \
#         #     arg.train_dataset_dir,arg.train_dataset_pattern,arg.model,arg.is_param_optim
#         # workers = arg.workers
#
#         enhancer_len, promoter_len, heads, num_layers, num_hiddens, ffn_num_hiddens = \
#             arg.enhancer_len,arg.promoter_len,arg.heads,arg.num_layers,arg.num_hiddens,arg.ffn_num_hiddens
#
#         encode_method, concat_reverse = arg.encode_method,arg.concat_reverse
#         # param_optim_strategy,k_fold,epochs, lr, optimizer, batch_size,  momentum = \
#         #     arg.param_optim_strategy,arg.k_fold,arg.epochs, arg.lr, arg.optimizer, arg.batch_size,  arg.momentum
#         # init = arg.init
#         device = arg.device
#         save_param_dir = arg.save_param_dir
#         # device, random_state, save_param_dir, save_model = arg.device, arg.random_state, arg.save_param_dir, arg.save_model
#         # params_config = arg.params_config
#         # random_size = arg.random_size
#         # init_points, n_iter = arg.init_points,arg.n_iter
#         ##
#         # weight_decay, nesterov, betas, eps, lr_decay, initial_accumulator_value, rho, alpha, \
#         #                  lambd, t0, max_iter, max_eval = arg.weight_decay, arg.nesterov, arg.betas, arg.eps, arg.lr_decay, arg.initial_accumulator_value, arg.rho, arg.alpha, \
#         #                  arg.lambd, arg.t0, arg.max_iter, arg.max_eval
#         save_param_prefix = arg.save_param_prefix
#         # rerun = arg.rerun
#         # save_fig = arg.save_fig
#         # metrics = arg.metrics
#         # log_dir = arg.log_dir
#         verbose = arg.verbose
# preprocess_module.add_argument("-s","--step",dest="step",default=None, help="choose specific step to run\n"
#                                            "    generate_interactions: generate valid interactions within compartment A/B\n"
#                                            "    filtration: filt the valid interactions based on user specified parameters\n"
#                                            "    balance: \n"
#                                            "    extract_seqs\n")
# dataset_name,
# concat_reverse, encode_method
# chrom_size_file,valid_pairs_file,eigenvec_dir,eigenvec_file_pattern,bin_size,keep_chrom_file=None,discard_chrom_file
# datasets_group = parser.add_argument_group("Datasets")
# datasets_group.add_argument('--lr', type=float, help='学习率')
# train_group = parser.add_argument_group("Train")
# train_group.add_argument('--lrs', type=float, help='学习率')
# evaluation_group = parser.add_argument_group("Evaluation")
# evaluation_group = parser.add_argument_group("Prediction")
# # preprocess_group.add_argument('--lr', type=float, help='学习率')
# parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",default=True,
#                     help="output gff3 file")
arg = parser.parse_args()


def main():
    module = arg.module
    ### Global configs

    ### Preprocess
    if module == "Preprocess":
        temp_dir = arg.temp_dir
        output = arg.output
        chunksize = arg.chunksize
        ## generate_compartments
        step, hic_file, juicer_tools_path, threads, chrom_size_file = \
            arg.step, arg.hic_file, arg.juicer_tools_path, arg.threads, arg.chrom_size_file
        method, flag, bin_size, output_dir, output_prefix, verbose = \
            arg.method, arg.flag, arg.bin_size, arg.output_dir, arg.output_prefix, arg.verbose

        ##
        chrom_size_file, valid_pairs_file, eigenvec_dir, eigenvec_file_pattern, keep_chrom_file, discard_chrom_file = \
            arg.chrom_size_file,arg.input,arg.eigenvec_dir,arg.eigenvec_file_pattern,arg.keep_chrom_file,arg.discard_chrom_file
        if output_dir is None:
            output_dir = os.path.dirname(output)
        check_and_create_dir(output_dir)
        progress_file = os.path.join(output_dir, "progress.pkl")

        ## Filtration
        input_interaction = arg.input
        interact_type, promoter, distance, within_compartment, annotation = arg.interact_type,arg.promoter,arg.distance,arg.within_compartment,arg.annotation
        promoter_range = arg.promoter_range
        distance_type = arg.distance_type
        promoter_number = arg.promoter_number
        length = arg.length
        output_raw = arg.output_raw

        ##
        filter_interactions_file, raw_interactions_file, reference_genome, GC_content = arg.input, arg.raw_interactions_file,arg.reference_genome, arg.GC_content
        balanced_interaction_type, balanced_same_chromsome = arg.balanced_interaction_type,arg.balanced_same_chromsome

        ##
        raw_interaction_fasta_file = arg.raw_interaction_fasta_file
        concat_reverse, encode_method = arg.concat_reverse, arg.encode_method
        balanced_interaction_file = arg.input
        dataset_name = arg.dataset_name
        # equal_length = arg.equal_length
        promoter_length = arg.promoter_length
        enhancer_length = arg.enhancer_length
        concat_epi = arg.concat_epi
        trim, padding = arg.trim, arg.padding
        compression = arg.compression
        pretrained_vec_file, k_mer = arg.pretrained_vec_file, arg.k_mer

    ### Datasets module

    # print(split)
    if module == "Preprocess":
        progress_key = ["generate_compartments","generate_interactions","filtration","balance","extract_dataset"]
        fprint_step(module)
        if step is None:

            progress_tracker = progress_logger.ProgressTracker(progress_file=progress_file,progress_key=progress_key)
            print(progress_tracker.progress)
            if not progress_tracker.get_progress(progress_key[0]):
                fprint_step(step=progress_key[0])
                generateCompartments = generate_compartments.GenerateCompartments(juicer_tools_path, hic_file, threads,
                                                                                  chrom_size_file,
                                                                                  method, flag,
                                                                                  bin_size, output_dir, output_prefix, verbose)
                try:
                    generateCompartments.run()
                    progress_tracker.update_progress(progress_key[0], True)
                except Exception as e:
                    # print(e)
                    traceback.print_exc()
                    progress_tracker.update_progress(progress_key[0], False)
                    sys.exit(-1)
            if not progress_tracker.get_progress(progress_key[1]):

                output_generate_interactions = os.path.join(output_dir, "sample/annotated_interactions.txt")
                generateInteractions = generate_interactions.GenerateInteractions(chrom_size_file, valid_pairs_file,temp_dir,
                                                                                  eigenvec_dir, eigenvec_file_pattern,
                                                                                  bin_size, output_generate_interactions,threads,chunksize, verbose,keep_chrom_file,
                                                                                  discard_chrom_file)
                # generateInteractions.run()
                fprint_step(step=progress_key[1])
                try:
                    generateInteractions.run()
                    # progress_tracker.update_progress("generate_interactions", )
                    progress_tracker.update_progress(progress_key[1],True,{"output_generate_interactions":output_generate_interactions})
                except Exception as e:
                    # print(e)
                    traceback.print_exc()

                    progress_tracker.update_progress(progress_key[1], False)
                    sys.exit(-1)
            if not progress_tracker.get_progress(progress_key[2]):
                fprint_step(step=progress_key[2])
                data = progress_tracker.get_progress_data(progress_key[1])
                input_interaction = data["output_generate_interactions"]
                generateInteractions = filtration.FilterInteractions(input_interaction, interact_type, promoter,
                                                                     distance, within_compartment, threads, chunksize,chrom_size_file,
                                                                     annotation, promoter_range,promoter_number,distance_type,length,
                                                                     output, temp_dir,output_raw,
                                                                     verbose)
                # input_interaction =
                try:
                    generateInteractions.run()
                    progress_tracker.update_progress(progress_key[2], True)
                except Exception as e:
                    # print(e)
                    traceback.print_exc()
                    progress_tracker.update_progress(progress_key[2], False)
                    sys.exit(-1)
            if not progress_tracker.get_progress(progress_key[3]):
                fprint_step(step="balance")
                balance_datasets = balance.BalanceDatasets(filter_interactions_file, raw_interactions_file,
                                                           reference_genome, GC_content,
                                                           balanced_interaction_type, balanced_same_chromsome,
                                                           threads, chunksize, temp_dir,
                                                           output, verbose)
                try:
                    balance_datasets.run()
                    progress_tracker.update_progress(progress_key[3], True)
                except Exception as e:
                    traceback.print_exc()
                    progress_tracker.update_progress(progress_key[3], False)
                    sys.exit(-1)
            if not progress_tracker.get_progress(progress_key[4]):
                fprint_step(step="extract_dataset")
                extract_datasets = extract_seqs.ExtractDatasets(balanced_interaction_file, raw_interaction_fasta_file,dataset_name,
                                                                concat_reverse,enhancer_length,promoter_length,trim, padding,compression,concat_epi,
                                                                pretrained_vec_file, k_mer,
                                                                encode_method, temp_dir, output_dir, output, threads,
                                                                chunksize, verbose)
                try:
                    extract_datasets.run()
                    progress_tracker.update_progress(progress_key[4], True)
                except Exception as e:
                    traceback.print_exc()
                    progress_tracker.update_progress(progress_key[4], False)
                    sys.exit(-1)
            # pass

        elif step == "generate_compartments":
            fprint_step(step="generate_compartments")
            generateCompartments = generate_compartments.GenerateCompartments(juicer_tools_path, hic_file, threads, chrom_size_file,
                                                                          method, flag,
                                                                          bin_size, output_dir, output_prefix, verbose)
            generateCompartments.run()
        elif step == "generate_interactions":
            fprint_step(step="generate_interactions")
            generateInteractions = generate_interactions.GenerateInteractions(chrom_size_file,valid_pairs_file,temp_dir,eigenvec_dir,eigenvec_file_pattern,
                                                                              bin_size,output,threads, chunksize,verbose,keep_chrom_file,discard_chrom_file)
            generateInteractions.run()
            pass
        elif step == "filtration":
            fprint_step(step="filtration")
            # input_interaction =
            generateInteractions = filtration.FilterInteractions(input_interaction, interact_type, promoter, distance, within_compartment, threads, chunksize,chrom_size_file,
                                                                     annotation,promoter_range,promoter_number,distance_type,length,
                                                                     output, temp_dir,output_raw,
                                                                     verbose)
            generateInteractions.run()
            pass
        elif step == "balance":
            fprint_step(step="balance")
            balance_datasets = balance.BalanceDatasets(filter_interactions_file, raw_interactions_file,reference_genome,
                                                       GC_content, balanced_interaction_type,balanced_same_chromsome,
                                                       threads,  chunksize, temp_dir, output,verbose)

            balance_datasets.run()
        elif step == "extract_dataset":
            fprint_step(step="extract_dataset")
            extract_datasets = extract_seqs.ExtractDatasets(balanced_interaction_file, raw_interaction_fasta_file, dataset_name, concat_reverse,
                                                            enhancer_length,promoter_length, trim, padding,compression,concat_epi,pretrained_vec_file,k_mer,
                                                            encode_method,temp_dir, output_dir, output, threads, chunksize, verbose)

            extract_datasets.run()
        else:
            fprint("Error","Wrong step specified!")

    elif module == "Datasets":
        # split = arg.split
        input_file = arg.input
        key = arg.key
        chunk_size = arg.chunksize
        test_size = arg.test_size
        output_format = arg.output_format
        random_state = arg.random_state
        split = arg.split
        output_prefix = arg.output_prefix
        compression = arg.compression

        verbose = arg.verbose

        prep_datasets = PrepDatasets(input_file, key, test_size, split, chunk_size, output_format, output_prefix, compression,
                                     random_state, verbose)
        prep_datasets.run()
        # try:
        #     prep_datasets = PrepDatasets(input_file,key,test_size,split,output_format,output_prefix,compression,random_state,verbose)
        #     prep_datasets.run()
        # except Exception as e:
        #     print(e)
        #     sys.exit(-1)
        #     pass

        pass

    elif module == "Train":
        train_dataset_file = arg.input
        test_dataset_file = arg.test_input
        train_dataset_dir,train_dataset_pattern,model,is_param_optim = \
            arg.train_dataset_dir,arg.train_dataset_pattern,arg.model,arg.is_param_optim
        workers = arg.workers

        enhancer_len, promoter_len, heads, num_layers, num_hiddens, ffn_num_hiddens = \
            arg.enhancer_len,arg.promoter_len,arg.heads,arg.num_layers,arg.num_hiddens,arg.ffn_num_hiddens

        encode_method, concat_reverse = arg.encode_method,arg.concat_reverse
        param_optim_strategy,k_fold,epochs, lr, optimizer, batch_size,  momentum = \
            arg.param_optim_strategy,arg.k_fold,arg.epochs, arg.lr, arg.optimizer, arg.batch_size,  arg.momentum
        init = arg.init
        device, random_state, save_param_dir, save_model = arg.device, arg.random_state, arg.save_param_dir, arg.save_model
        params_config = arg.params_config
        random_size = arg.random_size
        init_points, n_iter = arg.init_points,arg.n_iter
        ##
        weight_decay, nesterov, betas, eps, lr_decay, initial_accumulator_value, rho, alpha, \
                         lambd, t0, max_iter, max_eval = arg.weight_decay, arg.nesterov, arg.betas, arg.eps, arg.lr_decay, arg.initial_accumulator_value, arg.rho, arg.alpha, \
                         arg.lambd, arg.t0, arg.max_iter, arg.max_eval
        save_param_prefix = arg.save_param_prefix
        rerun = arg.rerun
        save_fig = arg.save_fig
        metrics = arg.metrics
        log_dir = arg.log_dir
        verbose = arg.verbose

        master = arg.master
        nodes = arg.nodes
        gpus = arg.gpus
        nr = arg.nr
        ddp_info = [nodes,gpus,nr,master]
        # print(master.split(":")[0])
        if master:
            os.environ['MASTER_ADDR'] = master.split(":")[0]
            os.environ['MASTER_PORT'] = master.split(":")[1]

        train_model = TrainModel(ddp_info,train_dataset_dir, train_dataset_pattern, train_dataset_file,test_dataset_file, workers,
                                 model, encode_method, concat_reverse,
                                 enhancer_len, promoter_len, heads, num_layers, num_hiddens, ffn_num_hiddens,
                                 is_param_optim,
                                 param_optim_strategy, params_config,random_size,init_points, n_iter,
                                 k_fold,metrics,save_fig,init,
                                 epochs, lr, optimizer, batch_size, momentum,
                                 weight_decay, nesterov, betas, eps, lr_decay, initial_accumulator_value, rho, alpha, \
                                 lambd, t0, max_iter, max_eval,
                                 device, random_state,
                                 rerun, save_param_dir, save_param_prefix,log_dir, save_model, verbose)
        train_model.run()
        # try:
        #     train_model = TrainModel(train_dataset_dir, train_dataset_pattern, train_dataset_file, model, is_param_optim, param_optim_strategy, k_fold,
        #          epochs, lr, optimizer, batch_size,momentum,device,random_state,
        #          save_param, save_model, verbose)
        #     train_model.run()
        # except Exception as e:
        #     print(e)
        #     sys.exit(-1)

    elif module == "Predict":
        enhancer_file = arg.enhancer_file
        promoter_file = arg.promoter_file
        enhancer_seq,promoter_seq = arg.enhancer_seq, arg.promoter_seq
        model = arg.model
        concat_epi, trim, padding, k_mer, pretrained_vec_file, threads = arg.concat_epi,arg.trim, arg.padding,arg.k_mer,arg.pretrained_vec_file,arg.threads
        # train_dataset_dir,train_dataset_pattern,model,is_param_optim = \
        #     arg.train_dataset_dir,arg.train_dataset_pattern,arg.model,arg.is_param_optim
        # workers = arg.workers

        enhancer_len, promoter_len, heads, num_layers, num_hiddens, ffn_num_hiddens = \
            arg.enhancer_len,arg.promoter_len,arg.heads,arg.num_layers,arg.num_hiddens,arg.ffn_num_hiddens

        encode_method, concat_reverse = arg.encode_method,arg.concat_reverse
        # param_optim_strategy,k_fold,epochs, lr, optimizer, batch_size,  momentum = \
        #     arg.param_optim_strategy,arg.k_fold,arg.epochs, arg.lr, arg.optimizer, arg.batch_size,  arg.momentum
        # init = arg.init
        device = arg.device
        save_param_dir = arg.save_param_dir
        # device, random_state, save_param_dir, save_model = arg.device, arg.random_state, arg.save_param_dir, arg.save_model
        # params_config = arg.params_config
        # random_size = arg.random_size
        # init_points, n_iter = arg.init_points,arg.n_iter
        ##
        # weight_decay, nesterov, betas, eps, lr_decay, initial_accumulator_value, rho, alpha, \
        #                  lambd, t0, max_iter, max_eval = arg.weight_decay, arg.nesterov, arg.betas, arg.eps, arg.lr_decay, arg.initial_accumulator_value, arg.rho, arg.alpha, \
        #                  arg.lambd, arg.t0, arg.max_iter, arg.max_eval
        save_param_prefix = arg.save_param_prefix
        # rerun = arg.rerun
        # save_fig = arg.save_fig
        # metrics = arg.metrics
        # log_dir = arg.log_dir
        verbose = arg.verbose

        predict_model = PredictEPI(model,
                 encode_method,concat_reverse,enhancer_len,promoter_len,heads,num_layers,num_hiddens,ffn_num_hiddens,
                 concat_epi,trim, padding,k_mer,pretrained_vec_file,threads,
                 save_param_dir, save_param_prefix,enhancer_seq,promoter_seq,enhancer_file,promoter_file,device,verbose)
        predict_model.run()
        # try:
        #     train_model = TrainModel(train_dataset_dir, train_dataset_pattern, train_dataset_file, model, is_param_optim, param_optim_strategy, k_fold,
        #          epochs, lr, optimizer, batch_size,momentum,device,random_state,
        #          save_param, save_model, verbose)
        #     train_model.run()
        # except Exception as e:
        #     print(e)
        #     sys.exit(-1)
    pass


if __name__ == '__main__':
    main()